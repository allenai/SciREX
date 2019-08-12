// Library that accepts a parameter dict and returns a full config.

function(p) {
  // Location of ACE valid event configs
  local valid_events_dir = "/data/dave/proj/dygie/dygie-experiments/datasets/ace-event/valid-configurations",

  // Storing constants.

  local validation_metrics = {
    "ner": "+ner_f1",
    "rel": "+rel_f1",
    "coref": "+coref_f1"
  },

  local display_metrics = {
    "ner": ["ner_precision", "ner_recall", "ner_f1"],
    "rel": ["rel_precision", "rel_recall", "rel_f1", "rel_span_recall"],
    "coref": ["coref_precision", "coref_recall", "coref_f1", "coref_mention_recall"]
  },

  local glove_dim = 300,
  local elmo_dim = 1024,

  local module_initializer = [
    [".*linear_layers.*weight", {"type": "xavier_normal"}],
    [".*scorer._module.weight", {"type": "xavier_normal"}],
    ["_distance_embedding.weight", {"type": "xavier_normal"}]],

  local dygie_initializer = [
    ["_span_width_embedding.weight", {"type": "xavier_normal"}],
    ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
    ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
  ],


  ////////////////////////////////////////////////////////////////////////////////

  // Calculating dimensions.

  local token_embedding_dim = ((if p.use_glove then glove_dim else 0) +
    (if p.use_char then p.char_n_filters else 0) +
    (if p.use_elmo then elmo_dim else 0)),
  local endpoint_span_emb_dim = 4 * p.lstm_hidden_size + p.feature_size,
  local attended_span_emb_dim = if p.use_attentive_span_extractor then token_embedding_dim else 0,
  local span_emb_dim = endpoint_span_emb_dim + attended_span_emb_dim,
  local pair_emb_dim = 3 * span_emb_dim,
  local relation_scorer_dim = pair_emb_dim,
  local coref_scorer_dim = pair_emb_dim + p.feature_size,
  local trigger_scorer_dim = 2 * p.lstm_hidden_size,  // Triggers are single contextualized tokens.
  local argument_scorer_dim = trigger_scorer_dim + span_emb_dim, // Trigger embeddings  and span embeddings.

  // Co-training
  local co_train = if "co_train" in p then p.co_train else false,

  ////////////////////////////////////////////////////////////////////////////////

  // Function definitions

  local make_feedforward(input_dim) = {
    input_dim: input_dim,
    num_layers: p.feedforward_layers,
    hidden_dims: p.feedforward_dim,
    activations: "relu",
    dropout: p.feedforward_dropout
  },

  // Model components

  local token_indexers = {
    [if p.use_glove then "tokens"]: {
      type: "single_id",
      lowercase_tokens: false
    },
    [if p.use_char then "token_characters"]: {
      type: "characters",
      min_padding_length: 5
    },
    [if p.use_elmo then "elmo"]: {
      type: "elmo_characters"
    }
  },

  local text_field_embedder = {
    token_embedders: {
      [if p.use_glove then "tokens"]: {
        type: "embedding",
        pretrained_file: if p.debug then null else "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
        embedding_dim: 300,
        trainable: false
      },
      [if p.use_char then "token_characters"]: {
        type: "character_encoding",
        embedding: {
          num_embeddings: 262,
          embedding_dim: 16
        },
        encoder: {
          type: "cnn",
          embedding_dim: 16,
          num_filters: p.char_n_filters,
          ngram_filter_sizes: [5]
        }
      },
      [if p.use_elmo then "elmo"]: {
        type: "elmo_token_embedder",
        options_file: "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        weight_file: "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
        do_layer_norm: false,
        dropout: 0.5
      }
    }
  },

  ////////////////////////////////////////////////////////////////////////////////

  // The model

  dataset_reader: {
    type: "ie_json",
    token_indexers: token_indexers,
    max_span_width: p.max_span_width
  },
  train_data_path: std.extVar("ie_train_data_path"),
  validation_data_path: std.extVar("ie_dev_data_path"),
  test_data_path: std.extVar("ie_test_data_path"),
  model: {
    type: "dygie",
    text_field_embedder: text_field_embedder,
    initializer: dygie_initializer,
    loss_weights: p.loss_weights,
    lexical_dropout: p.lexical_dropout,
    lstm_dropout: p.lstm_dropout,
    feature_size: p.feature_size,
    use_attentive_span_extractor: p.use_attentive_span_extractor,
    max_span_width: p.max_span_width,
    display_metrics: display_metrics[p.target],
    valid_events_dir: valid_events_dir,
    co_train: co_train,
    context_layer: {
      type: "lstm",
      bidirectional: true,
      input_size: token_embedding_dim,
      hidden_size: p.lstm_hidden_size,
      num_layers: p.lstm_n_layers,
      dropout: p.lstm_dropout
    },
    modules: {
      coref: {
        spans_per_word: p.coref_spans_per_word,
        max_antecedents: p.coref_max_antecedents,
        mention_feedforward: make_feedforward(span_emb_dim),
        antecedent_feedforward: make_feedforward(coref_scorer_dim),
        initializer: module_initializer
      },
      ner: {
        mention_feedforward: make_feedforward(span_emb_dim),
        initializer: module_initializer
      },
      relation: {
        spans_per_word: p.relation_spans_per_word,
        positive_label_weight: p.relation_positive_label_weight,
        mention_feedforward: make_feedforward(span_emb_dim),
        relation_feedforward: make_feedforward(relation_scorer_dim),
        initializer: module_initializer
      },
      events: {
        trigger_spans_per_word: p.trigger_spans_per_word,
        argument_spans_per_word: p.argument_spans_per_word,
        positive_label_weight: p.events_positive_label_weight,
        trigger_feedforward: make_feedforward(trigger_scorer_dim),
        trigger_candidate_feedforward: make_feedforward(trigger_scorer_dim),
        mention_feedforward: make_feedforward(span_emb_dim),
        argument_feedforward: make_feedforward(argument_scorer_dim),
        initializer: module_initializer,
        loss_weights: p.loss_weights_events
      }
    }
  },
  iterator: {
    type: if co_train then "ie_multitask" else "ie_batch",
    batch_size: p.batch_size
  },
  validation_iterator: {
    type: "ie_document",
  },
  trainer: {
    num_epochs: p.num_epochs,
    grad_norm: 5.0,
    patience : p.patience,
    cuda_device : std.parseInt(std.extVar("cuda_device")),
    validation_metric: validation_metrics[p.target],
    learning_rate_scheduler: p.learning_rate_scheduler,
    optimizer: p.optimizer
  }
}
