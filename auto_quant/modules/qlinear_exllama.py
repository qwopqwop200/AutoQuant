import gc
import math
from typing import Optional
import torch
import torch.nn as nn
from exllama_kernels import make_q4, q4_matmul, prepare_buffers, set_tuning_params, cleanup_buffers_cuda

EXLLAMA_DEFAULT_MAX_INPUT_LENGTH = 2048

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")

def ext_make_q4(qweight, qzeros, scales, g_idx, device):
    """Construct Q4Matrix, return handle"""
    return make_q4(qweight,
                   qzeros,
                   scales,
                   g_idx if g_idx is not None else none_tensor,
                   device)

def ext_q4_matmul(x, q4, q4_width):
    """Matrix multiplication, returns x @ q4"""
    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.float16, device=x.device)
    q4_matmul(x, q4, output)
    return output.view(outshape)

class ExllamaLinear(nn.Module):
    def __init__(self, bits, group_size, in_features, out_features, bias, device):
        super().__init__()
        if bits not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size != -1 else in_features
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.bits) == 0
        
        self.register_buffer('qweight',torch.zeros((in_features // 32 * bits), out_features, dtype=torch.int32, device=device))
        self.register_buffer('qzeros',torch.zeros((math.ceil(in_features / group_size), out_features // 32 * bits), dtype=torch.int32, device=device))
        self.register_buffer('scales',torch.zeros((math.ceil(in_features / group_size), out_features), dtype=torch.float16, device=device))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=device))
        else:
            self.bias = None
            
    def post_init(self):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        self.q4 = ext_make_q4(
            self.qweight,
            self.qzeros,
            self.scales,
            None,
            self.qweight.device.index # device index
        )
        
    @classmethod
    def from_linear(cls, linear, bits, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(bits, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear
        
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.bits
        
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[idx // group_size]) / awq_linear.scales[idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        
        qweight = torch.zeros((intweight.shape[0]  // 32 * awq_linear.bits, intweight.shape[1]), dtype=torch.int32, device=intweight.device)           
        for row in range(intweight.shape[0] // pack_num):
            for i in range(pack_num):
                qweight[row] |= intweight[row * pack_num + i] << (i * awq_linear.bits)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 32 * awq_linear.bits), dtype=torch.int32, device=zeros.device)
        for col in range(zeros.shape[1] // pack_num):     
            for i in range(pack_num):
                qzeros[:, col] |= zeros[:, col * pack_num + i] << (i * awq_linear.bits)
        awq_linear.qzeros = qzeros
        return awq_linear
        
    @torch.no_grad()
    def forward(self, x):
        out = ext_q4_matmul(x.half(), self.q4, self.out_features)
        out = out + self.bias if self.bias is not None else out
        return out
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, bits={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.bits, self.group_size
        )
        
def exllama_post_init(model, use_act_order: bool, max_input_length: Optional[int] = None):
    """
    The max_input_length argument is specific to the exllama backend, that requires to initialize a buffer temp_state.
    """
    device_to_buffers_size = {}

    model_uses_exllama = False
    for name, submodule in model.named_modules():
        if isinstance(submodule, ExllamaLinear):
            model_uses_exllama = True
            device = submodule.qweight.device
            if device not in device_to_buffers_size:
                device_to_buffers_size[device] = {
                    "max_dq_buffer_size": 1,
                    "max_inner_outer_dim": 1
                }
            
            if not use_act_order:
                submodule._use_act_order = False
            else:
                submodule._use_act_order = True

            device_to_buffers_size[device]["max_dq_buffer_size"] = max(device_to_buffers_size[device]["max_dq_buffer_size"], submodule.qweight.numel() * 8)

            if use_act_order:
                device_to_buffers_size[device]["max_inner_outer_dim"] = max(device_to_buffers_size[device]["max_inner_outer_dim"], submodule.infeatures, submodule.outfeatures)

    if model_uses_exllama:
        device_to_buffers = {}

        if use_act_order:
            if max_input_length is None:
                max_input_len = EXLLAMA_DEFAULT_MAX_INPUT_LENGTH
            else:
                max_input_len = max_input_length
        else:
            if max_input_length is not None:
                logger.info("Using exllama backend without act-order, the parameter max_input_length was set although not needed, it will be ignored.")
            max_input_len = 1

        for device, buffers_size in device_to_buffers_size.items():
            # The temp_state buffer is required to reorder X in the act-order case.
            # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
            device_to_buffers[device] = {
                "temp_state": torch.zeros((max_input_len, buffers_size["max_inner_outer_dim"]), dtype=torch.float16, device=device),
                "temp_dq": torch.zeros((1, buffers_size["max_dq_buffer_size"]), dtype=torch.float16, device=device),
                "max_dq_buffer_size": buffers_size["max_dq_buffer_size"],
                "max_inner_outer_dim": buffers_size["max_inner_outer_dim"],
            }
        
        # Buffers need to be persistent to avoid any bug.
        model.device_to_buffers = device_to_buffers
    
        for device, buffers in model.device_to_buffers.items():
            prepare_buffers(device, buffers["temp_state"], buffers["temp_dq"])

        # Using the default from exllama repo here.
        matmul_recons_thd = 8
        matmul_fused_remap = False
        matmul_no_half2 = False
        set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

        # The buffers need to have been initialized first before calling make_q4.
        for name, submodule in model.named_modules():
            if isinstance(submodule, ExllamaLinear):
                submodule.post_init()

        torch.cuda.empty_cache()
    
    return model
        
def exllama_set_max_input_length(model, max_input_length: int):
    """
    This method does not necessarily require `model` to inherit from BaseGPTQForCausalLM.

    When using the exllama backend with act-order, it is necessary to initialize a buffer that depends on the maximum expected input length. In case the
    default used (EXLLAMA_DEFAULT_MAX_INPUT_LENGTH) is too short, this method can be called to extend the buffer size without reloading the whole model.
    """
    if not model.quantize_config.desc_act:
        raise ValueError("The method exllama_set_max_input_length should be called only when using the exllama backend **with act-order**.")
    
    device_to_buffers_size = {}
    for device, buffers in model.device_to_buffers.items():
        device_to_buffers_size[device] = {"max_dq_buffer_size": buffers["max_dq_buffer_size"], "max_inner_outer_dim": buffers["max_inner_outer_dim"]}
    
    # For an unknown reason calling just `del model.device_to_buffers` raises an AttributeError.
    for key in list(model.device_to_buffers.keys()):
        del model.device_to_buffers[key]
    model.device_to_buffers = None
    del model.device_to_buffers

    gc.collect()
    torch.cuda.empty_cache()
    # cleanup_buffers_cuda()

    device_to_buffers = {}
    for device, buffers_size in device_to_buffers_size.items():
        # The temp_state buffer is required to reorder X in the act-order case.
        # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
        device_to_buffers[device] = {
            "temp_state": torch.zeros((max_input_length, buffers_size["max_inner_outer_dim"]), dtype=torch.float16, device=device),
            "temp_dq": torch.zeros((1, buffers_size["max_dq_buffer_size"]), dtype=torch.float16, device=device),
            "max_dq_buffer_size": buffers_size["max_dq_buffer_size"],
            "max_inner_outer_dim": buffers_size["max_inner_outer_dim"],
        }

        prepare_buffers(device, device_to_buffers[device]["temp_state"], device_to_buffers[device]["temp_dq"])

    # Buffers need to be persistent to avoid any bug.
    model.device_to_buffers = device_to_buffers

    return model
    
def make_sure_no_tensor_in_meta_device(model):
    for n, m in model.named_modules():
        if isinstance(submodule, ExllamaLinear) and m.bias.device == torch.device("meta"):
            m.register_buffer('bias', torch.zeros((m.outfeatures), dtype=torch.float16, device="cpu"))