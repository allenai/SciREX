// Library that accepts a parameter dict and returns a full config.

function(p) {
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
  local bert_base_dim = 768,
  local char_n_filters = 128,

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

  local use_bert = p.use_bert,
  local token_embedding_dim = (
    (if p.use_glove then glove_dim else 0) +
    (if p.use_char then char_n_filters else 0) +
    (if p.use_bert then bert_base_dim else 0)
  ),
  local endpoint_span_emb_dim = 4 * p.lstm_hidden_size + p.feature_size,
  local attended_span_emb_dim = if p.use_attentive_span_extractor then token_embedding_dim else 0,
  local span_emb_dim = endpoint_span_emb_dim + attended_span_emb_dim,
  local pair_emb_dim = 3 * span_emb_dim,
  local relation_scorer_dim = pair_emb_dim,
  local coref_scorer_dim = pair_emb_dim + p.feature_size,

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
    [if use_bert then "bert"]: {
      type: "bert-pretrained",
      pretrained_model: std.extVar("BERT_VOCAB"),
      do_lowercase: std.extVar("IS_LOWERCASE"),
      use_starting_offsets: true,
      truncate_long_sequences : true
    }
  },
  local text_field_embedder = {
      [if use_bert then "allow_unmatched_keys"]: true,
      [if use_bert then "embedder_to_indexer_map"]: {
        bert: ["bert", "bert-offsets"],
        token_characters: ["token_characters"],
        tokens: ["tokens"],
      },
      token_embedders : {
        [if p.use_glove then "tokens"]: {
          type: "embedding",
          pretrained_file: if p.debug then null else "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
          embedding_dim: 300,
          trainable: false
        },
        [if p.use_char then "token_characters"]: {
          type: "character_encoding",
          embedding: {
            embedding_dim: 16
          },
          encoder: {
            type: "cnn",
            embedding_dim: 16,
            num_filters: 128,
            ngram_filter_sizes: [5],
            conv_layer_activation: "relu"
          }
        },
        [if p.use_bert then "bert"]: {
            type: "bert-pretrained",
            pretrained_model: std.extVar("BERT_WEIGHTS"),
            //requires_grad: "11",
            //top_layer_only: false
        }
    }
  },

  ////////////////////////////////////////////////////////////////////////////////

  // The model

  dataset_reader: {
    type: "ie_json",
    token_indexers: token_indexers,
    max_span_width: p.max_span_width,
    context_width: p.context_width
  },
  train_data_path: std.extVar("TRAIN_PATH"),
  validation_data_path: std.extVar("DEV_PATH"),
  test_data_path: std.extVar("TEST_PATH"),
  //regularizers: {
  //should match every layer
  //  ["*"]: {
  //    "type": "l2",
  //    "alpha": 0.001,
  //  },
  //},
  model: {
    type: "dygie",
    text_field_embedder: text_field_embedder,
    initializer: dygie_initializer,
    loss_weights: p.loss_weights,
    lexical_dropout: p.lexical_dropout,
    feature_size: p.feature_size,
    use_attentive_span_extractor: p.use_attentive_span_extractor,
    max_span_width: p.max_span_width,
    display_metrics: display_metrics[p.target],
    context_layer: {
      type: "lstm",
      bidirectional: true,
      input_size: token_embedding_dim,
      hidden_size: p.lstm_hidden_size,
      num_layers: p.lstm_n_layers,
    },
    modules: {
      coref: {
        span_emb_dim: span_emb_dim,
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
        rel_prop_dropout_A: p.rel_prop_dropout_A,
        rel_prop_dropout_f: p.rel_prop_dropout_f,
        span_emb_dim: span_emb_dim,
        rel_prop: p.rel_prop,
        initializer: module_initializer
      }
    }
  },
  iterator: {
    type: "ie_batch",
    batch_size: p.batch_size
  },
  validation_iterator: {
    type: "ie_document",
  },
  trainer: {
    num_epochs: p.num_epochs,
    grad_norm: 5.0,
    patience : p.patience,
    cuda_device : std.parseInt(std.extVar("CUDA_DEVICE")),
    validation_metric: validation_metrics[p.target],
    learning_rate_scheduler: p.learning_rate_scheduler,
    optimizer: p.optimizer
  }
}
