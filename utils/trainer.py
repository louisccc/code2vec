class Trainer:
    ''' the trainer for code2vec '''
    def __init__(self, path):
        self.config = Config()

        self.reader = PathContextReader(path)
        self.reader.read_path_contexts()
        self.config.path = path

        self.config.num_of_words = len(self.reader.word_count)
        self.config.num_of_paths = len(self.reader.path_count)
        self.config.num_of_tags  = len(self.reader.target_count)

        self.model = code2vec(self.config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.config.learning_rate)
    
    @tf.function
    def train_step(self, e1, p, e2, tags):
        with tf.GradientTape() as tape:
            code_vectors, attention_weights = self.model.forward(e1, p, e2)
            logits = tf.matmul(code_vectors, self.model.tags_embeddings, transpose_b=True)
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(tags, [-1]), logits=logits))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss
    
    @tf.function
    def test_step(self, e1, p, e2, topk=1):
        code_vectors, _ = self.model.forward(e1, p, e2, train=False)
        logits = tf.matmul(code_vectors, self.model.tags_embeddings, transpose_b=True)
        _, ranks = tf.nn.top_k(logits, k=topk)
        return ranks

    def train_model(self):

        self.train_batch_generator = Generator(self.reader.bags_train, self.config.training_batch_size)
        
        for epoch_idx in tqdm(range(self.config.epoch)):
            
            acc_loss = 0

            for batch_idx in range(self.train_batch_generator.number_of_batch):
                
                data, tag = next(self.train_batch_generator)

                e1 = data[:,:,0]
                p  = data[:,:,1]
                e2 = data[:,:,2]
                y  = tag

                loss = self.train_step(e1, p, e2, y)

                acc_loss += loss

            if epoch_idx % 5 == 0:
                print("Evaluation Set Test:")
                self.evaluate_model(training_set=False)
                print("Training Set Test:")
                self.evaluate_model(training_set=True)
                self.save_model(epoch_idx)
                self.export_code_embeddings(epoch_idx)

            print('epoch[%d] ---Acc Train Loss: %.5f' % (epoch_idx, acc_loss))

    def evaluate_model(self, training_set = False):
        if training_set: 
            self.test_batch_generator = Generator(self.reader.bags_train, self.config.testing_batch_size)
        else:
            self.test_batch_generator = Generator(self.reader.bags_test, self.config.testing_batch_size)

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        prediction_rank = 0 
        prediction_reciprocal_rank = 0
        nr_predictions = 0

        for batch_idx in range(self.test_batch_generator.number_of_batch):

            data, tag = next(self.test_batch_generator)
            
            e1 = data[:,:,0]
            p  = data[:,:,1]
            e2 = data[:,:,2]
            y  = tag
            
            ranks = self.test_step(e1, p, e2)
            
            ranks_number = tf.where(tf.equal(self.test_step(e1, p, e2, topk=self.config.num_of_tags), tf.cast(tf.expand_dims(y,-1), dtype=tf.int32)))

            for idx, rank_number in enumerate(ranks_number.numpy().tolist()): 
                prediction_rank += (rank_number[1] + 1)
                prediction_reciprocal_rank += 1.0 / (rank_number[1] + 1)

            for idx, rank in enumerate(ranks.numpy().tolist()):
                nr_predictions += 1
                
                original_name = self.reader.idx2target[tag.tolist()[idx]]
                inferred_names = [self.reader.idx2target[target_idx] for target_idx in rank]

                original_subtokens = original_name.split('|')
                

                true_positive = 0
                false_positive = 0
                false_negative = 0

                for inferred_name in inferred_names:
                    inferred_subtokens = inferred_name.split('|')

                    true_positive += sum(1 for subtoken in inferred_subtokens if subtoken in original_subtokens)
                    false_positive += sum(1 for subtoken in inferred_subtokens if subtoken not in original_subtokens)
                    false_negative += sum(1 for subtoken in original_subtokens if subtoken not in inferred_subtokens)

                # if false_positive > 0:
                #     print(original_name)
                #     print(inferred_names)

                true_positives += true_positive
                false_positives += false_positive
                false_negatives += false_negative

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        prediction_rank /= nr_predictions
        prediction_reciprocal_rank /= nr_predictions

        print("\nPrecision: {}, Recall: {}, F1: {} Rank: {} Reciprocal_Rank: {}\n".format(precision, recall, f1, prediction_rank, prediction_reciprocal_rank))
    
    def export_code_embeddings(self, epoch_idx):
        save_path = self.config.path / ('epoch_%d' % epoch_idx)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(str(save_path / "code_labels.tsv"), 'w') as l_export_file:
            for label in self.reader.idx2target.values():
                l_export_file.write(label + "\n")

            parameter = self.model.tags_embeddings
            
            all_ids = list(range(0, int(parameter.shape[0])))
            stored_name = parameter.name.split(':')[0]

            if len(parameter.shape) == 2:
                all_embs = parameter.numpy()
                with open(str(save_path / ("%s.tsv" % stored_name)), 'w') as v_export_file:
                    for idx in all_ids:
                        v_export_file.write("\t".join([str(x) for x in all_embs[idx]]) + "\n")

    def save_model(self, epoch_idx):
        saved_path = self.config.path / ('epoch_%d' % epoch_idx)
        saved_path.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(saved_path / 'model.vec'))

    def load_model(self, path):
        if path.exists():
            self.model.load_weights(str(path / 'model.vec'))
