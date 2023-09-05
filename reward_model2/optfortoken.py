import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import OPTPreTrainedModel, OPTModel
from transformers.modeling_outputs import TokenClassifierOutput

class OPTForTokenClassification(OPTPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self._config = config
    self.num_labels = config.num_labels if config.num_labels else 4
    self.transformer = OPTModel(config)
    if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
      classifier_dropout = config.classifier_dropout
    elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
      classifier_dropout = config.hidden_dropout
    else:
      classifier_dropout = 0.1
    self.dropout = nn.Dropout(classifier_dropout)
    self.classifier = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

    # init weights
    self.init_weights()

  def forward(
        self,
        input_ids = None,
        past_key_values = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        ):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    hidden_states = transformer_outputs[0]
    hidden_states = self.dropout(hidden_states)
    logits = self.classifier(hidden_states)

    loss = None
    if labels is not None:
        labels = labels.to(logits.device)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
        output = (logits,) + transformer_outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=transformer_outputs.hidden_states,
            # attentions=transformer_outputs.attentions,
        )