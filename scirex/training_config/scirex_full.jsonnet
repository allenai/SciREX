// Import template file.

local template = import "template_full.libsonnet";

// Set options.

local params = {
  // Primary prediction target. Watch metrics associated with this target.
  target: "ner",
  use_lstm: std.extVar("USE_LSTM"),
  bert_fine_tune: std.extVar("BERT_FINE_TUNE"),
  document_filter: std.extVar('DOCUMENT_FILTER'),
  filter_to_salient: stringToBool(std.extVar('FTS')),
  loss_weights: {          // Loss weights for the modules.
    ner: std.extVar('nw'),
    relation: 0.0,
    coref: 0.0,
    linked: std.extVar('lw'),
    n_ary_relation: std.extVar('rw'),
    cluster_saliency: std.extVar('cw')
  },
  relation_cardinality: std.parseInt(std.extVar('relation_cardinality')),
};

template(params)