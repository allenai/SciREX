// Import template file.

local template = import "template_crf_n_ary.libsonnet";

////////////////////

local stringToBool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

// Set options.

local params = {
  // Primary prediction target. Watch metrics associated with this target.
  target: "ner",
  dataset_reader: 'pwc_json_crf',
  // If debugging, don't load expensive embedding files.
  debug: false,

  // Specifies the token-level features that will be created.
  use_glove: false,
  use_char: true,
  use_bert: true,
  use_lstm: stringToBool(std.extVar("USE_LSTM")),
  bert_fine_tune: std.extVar("BERT_FINE_TUNE"),
  // Specifies the model parameters.
  lstm_hidden_size: 200,
  lstm_n_layers: 1,
  feedforward_layers: 2,
  feedforward_dim: 150,
  feedforward_dropout: 0.0,
  lexical_dropout: 0.0,
  lstm_dropout: 0.0,
  loss_weights: {          // Loss weights for the modules.
    ner: 0.0,
    relation: 0.0,
    coref: 0.0,
    linked: 0.0,
    n_ary_relation: 1.0
  },

  label_namespace: "ner_entity_labels",

  // Model training
  batch_size: 60,
  num_epochs: 100,
  shuffle_instances: false,
  optimizer: {
    type: "adam",
    lr: 1e-3
  },
  learning_rate_scheduler:  {
    type: "reduce_on_plateau",
    factor: 0.5,
    mode: "max",
    patience: 20
  }
};

////////////////////

// Feed options into template.

template(params)
