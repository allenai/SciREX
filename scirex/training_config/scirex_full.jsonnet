// Import template file.

local template = import "template_full.libsonnet";

// Set options.

local params = {
  // Primary prediction target. Watch metrics associated with this target.
  target: "ner",
  use_lstm: true,
  bert_fine_tune: std.extVar("bert_fine_tune"),
  loss_weights: {          // Loss weights for the modules.
    ner: std.extVar('nw'),
    saliency: std.extVar('lw'),
    n_ary_relation: std.extVar('rw')
  },
  relation_cardinality: std.parseInt(std.extVar('relation_cardinality')),
};

template(params)