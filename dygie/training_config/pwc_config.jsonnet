// Import template file.

local template = import "template.libsonnet";

local stringToBool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

////////////////////

// Set options.

local params = {
  // Primary prediction target. Watch metrics associated with this target.
  target: "ner",
  dataset_reader: 'pwc_json',
  debug: false,

  // Specifies the token-level features that will be created.
  use_glove: false,
  use_char: true,
  use_attentive_span_extractor: true,
  use_bert: true,
  use_lstm: stringToBool(std.extVar("USE_LSTM")),
  bert_fine_tune: std.extVar("BERT_FINE_TUNE"),
 
  // Specifies the model parameters.
  lstm_hidden_size: 200,
  lstm_n_layers: 1,
  feature_size: 20,
  feedforward_layers: 2,
  feedforward_dim: 150,
  max_span_width: 8,
  feedforward_dropout: 0.4,
  lexical_dropout: 0.5,
  balancing_strategy: std.extVar('BALANCING_STRATEGY'),
  lstm_dropout: 0.4,
  loss_weights: {          // Loss weights for the modules.
    ner: 1.0,
    relation: 0.0,
    coref: 0.0
  },

  // Coref settings.
  coref_spans_per_word: 0.1,
  coref_max_antecedents: 100,

  // Relation settings.
  relation_spans_per_word: 0.5,
  relation_positive_label_weight: 1.0,

  // Model training
  decoding_type: "all_decode",
  batch_size: 10,
  num_epochs: 100,
  shuffle_instances: true,
  patience: 8,
  optimizer: {
    type: "sgd",
    lr: 0.01,
    momentum: 0.9,
    nesterov: true,
  },
  learning_rate_scheduler:  {
    type: "reduce_on_plateau",
    factor: 0.5,
    mode: "max",
    patience: 8
  }
};

////////////////////

// Feed options into template.

template(params)
