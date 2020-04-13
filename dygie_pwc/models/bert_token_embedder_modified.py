from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, BertModel

from typing import List

@TokenEmbedder.register("bert-pretrained-modified")
class PretrainedBertEmbedder(BertEmbedder):

    """
    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .tar.gz file with the model weights.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L41
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    requires_grad : ``bool``, optional (default = False)
        If True, compute gradient of BERT parameters for fine tuning.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    scalar_mix_parameters: ``List[float]``, optional, (default = None)
        If not ``None``, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training.
    """

    def __init__(
        self,
        pretrained_model: str,
        requires_grad: str = "none",
        top_layer_only: bool = False,
        scalar_mix_parameters: List[float] = None,
        set_untrained_to_eval:bool = True
    ) -> None:
        model = BertModel.from_pretrained(pretrained_model)

        self._grad_layers = requires_grad
        self._set_untrained_to_eval = set_untrained_to_eval

        if requires_grad in ["none", "all"]:
            for param in model.parameters():
                param.requires_grad = requires_grad == "all"
        else:
            model_name_regexes = requires_grad.split(",")
            for name, param in model.named_parameters():
                found = False
                for regex in model_name_regexes:
                    if regex in name:
                        found = True
                        break
                param.requires_grad = found


        super().__init__(
            bert_model=model,
            top_layer_only=top_layer_only,
            scalar_mix_parameters=scalar_mix_parameters,
        )

    def train(self, mode=True):
        r"""
        Modify the normal .train method to set frozen layers to eval mode
        
        Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """

        self.training = mode
        for mod_name, module in self.named_modules() :
            if module == self :
                continue
            
            if 'bert_model' not in mod_name :
                module.train(mode)
                continue

            if self._grad_layers in ['none', 'all'] :
                module_requires_grad = self._grad_layers == 'all'
            else :
                model_name_regexes = self._grad_layers.split(",")
                module_requires_grad = any([regex in mod_name for regex in model_name_regexes])

            if self._set_untrained_to_eval and not module_requires_grad:
                module.eval()
            else :
                module.train(mode)

        return self