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
  rel_prop: 0,
  context_width: 3,
  // Specifies the model parameters.
  lstm_hidden_size: 200,
  lstm_n_layers: 1,
  feedforward_layers: 2,
  feedforward_dim: 150,
  feedforward_dropout: 0.4,
  lexical_dropout: 0.5,
  lstm_dropout: 0.4,
  loss_weights: {          // Loss weights for the modules.
    ner: 1.0,
    relation: 1.0,
    coref: 0.0
  },

  label_namespace: "ner_entity_labels",
  // Coref settings.
  coref_spans_per_word: 0.1,
  coref_max_antecedents: 100,

  // Relation settings.
  relation_positive_label_weight: 1.0,

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
    patience: 10
  }
};

////////////////////

// Feed options into template.

template(params)
