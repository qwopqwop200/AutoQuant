import logging
import torch
import transformers
from lm_eval.base import BaseLM

class LMEvalAdaptor(BaseLM):
    def __init__(self, model, tokenizer, batch_size=1, max_length=-1):
        super().__init__()
        assert isinstance(batch_size, int)
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = batch_size
        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length != -1:
            return self._max_length
        if hasattr(self.model.config, 'n_ctx'):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'n_positions'):
            return self.model.config.n_positions
        else:
            logger.warning("can't get model's sequence length from model config, will set to 2048.")
            return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            if isinstance(self.model, transformers.models.t5.modeling_t5.T5ForConditionalGeneration):
                decoder_start_token_id = self.model.generation_config.decoder_start_token_id
                dec_inps = torch.cat([torch.tensor(decoder_start_token_id,).tile(len(inps), 1).to(inps),inps,],dim=1,)
             
                kwargs = {"decoder_input_ids": dec_inps,}
            else:
                kwargs = {}
            out = self.model(inps, **kwargs)[0]
            if self.model.config.model_type == "opt":  # there are a few extra tokens in opt, which we should omit
                return out[:, :, :50257]
            else:
                return out  # [:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )

