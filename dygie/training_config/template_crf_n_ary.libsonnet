// Library that accepts a parameter dict and returns a full config.

function(p) {
  // Storing constants.

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
    char_n_filters +
    (if p.use_bert then bert_base_dim else 0)
  ),
  local context_encoder_dim = if p.use_lstm then 2 * p.lstm_hidden_size else token_embedding_dim,
  local endpoint_span_emb_dim = 2 * context_encoder_dim,
  local attended_span_emb_dim = context_encoder_dim,
  local span_emb_dim = endpoint_span_emb_dim + attended_span_emb_dim,
  local n_features = 1 + 4 + 5,
  local featured_emb_dim = span_emb_dim + n_features,

  ////////////////////////////////////////////////////////////////////////////////

  // Function definitions
  local lstm_context_encoder = {
    type: "lstm",
    bidirectional: true,
    input_size: token_embedding_dim,
    hidden_size: p.lstm_hidden_size,
    num_layers: p.lstm_n_layers
  },

  local pass_through_encoder = {
    type: "pass_through",
    input_dim: token_embedding_dim,
  },

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
      truncate_long_sequences : false
    }
  },
  local text_field_embedder = {
      [if use_bert then "allow_unmatched_keys"]: true,
      [if use_bert then "embedder_to_indexer_map"]: {
        bert: ["bert", "bert-offsets"],
        tokens: ["tokens"],
        token_characters: ["token_characters"]
      },
      token_embedders : {
        [if p.use_glove then "tokens"]: {
          type: "embedding",
          pretrained_file: if p.debug then null else "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
          embedding_dim: 300,
          trainable: false
        },
        [if p.use_bert then "bert"]: {
            type: "bert-pretrained-modified",
            pretrained_model: std.extVar("BERT_WEIGHTS"),
            requires_grad: p.bert_fine_tune,
            set_untrained_to_eval : p.set_to_eval
        },
        "token_characters": {
          type: "character_encoding",
          embedding: {
            embedding_dim: 16
          },
          encoder: {
            type: "cnn",
            embedding_dim: 16,
            num_filters: char_n_filters,
            ngram_filter_sizes: [5],
            conv_layer_activation: "relu"
          }
        },
    }
  },

  ////////////////////////////////////////////////////////////////////////////////

  // The model

  dataset_reader: {
    type: p.dataset_reader,
    token_indexers: token_indexers,
    document_filter_type: p.document_filter
  },
  train_data_path: std.extVar("TRAIN_PATH"),
  validation_data_path: std.extVar("DEV_PATH"),
  test_data_path: std.extVar("TEST_PATH"),

  model: {
    type: "dygie_crf_n_ary",
    text_field_embedder: text_field_embedder,
    initializer: dygie_initializer,
    loss_weights: p.loss_weights,
    lexical_dropout: p.lexical_dropout,
    display_metrics: ["validation_metric"],
    context_layer: if p.use_lstm then lstm_context_encoder else pass_through_encoder,
    modules: {
      coref: {
        antecedent_feedforward: make_feedforward(featured_emb_dim),
        initializer: module_initializer
      },
      ner: {
        mention_feedforward: make_feedforward(context_encoder_dim),
        label_namespace: p.label_namespace,
        label_encoding: 'BIOUL',
        initializer: module_initializer,
      },
      relation: {
        antecedent_feedforward: make_feedforward(3*featured_emb_dim),
        initializer: module_initializer
      },
      link_classifier: {
        mention_feedforward: make_feedforward(featured_emb_dim),
        label_namespace: "span_link_labels",
        initializer: module_initializer,
        n_features: n_features
      },
      n_ary_relation: {
        antecedent_feedforward: make_feedforward(p.relation_cardinality*featured_emb_dim),
        initializer: module_initializer,
	      relation_cardinality: p.relation_cardinality
      },
    }
  },
  iterator: {
    type: "ie_batch",
    batch_size: p.batch_size,
  },
  validation_iterator: {
    type: "ie_batch",
    batch_size: p.batch_size,
  },
  trainer: {
    num_epochs: p.num_epochs,
    grad_norm: 5.0,
    patience : 10,
    cuda_device : std.parseInt(std.extVar("CUDA_DEVICE")),
    validation_metric: '+validation_metric',
    learning_rate_scheduler: p.learning_rate_scheduler,
    optimizer: p.optimizer,
    num_serialized_models_to_keep: 1,
    should_log_learning_rate: true
  }
}
