import json 
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch 
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import pickle 
import os 
import re
import spacy_alignments as tokenizations
import itertools
import argparse



def save_json(path, data):
    with open(path, "w+") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return

def open_json(path):
    f = open(path)
    return json.load(f)

class TrainClassifier():

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model 
        return
    
    def pickle_classifier(self, classifier, experiment_name, data_config, layer, caseless, model):
        os.makedirs(os.path.dirname(f"./classifiers/{experiment_name}/{model}/caseless_{caseless}/{data_config}/{layer}_classifier.pkl"), exist_ok=True)
        with open(f"./classifiers/{experiment_name}/{model}/caseless_{caseless}/{data_config}/{layer}_classifier.pkl", 'wb') as f:
            pickle.dump(classifier,f)
        return 
    
    def get_token_ids(self, sent, mbert_ids, bert_tokenised_text):
        """
        align tokens from the mbert token ids 
        """
        tokenized_text, tokens_tensor, segments_tensor = self.mbert_text_preparation(sent)
        mbert_tokens = [re.sub("##", "", i) for i in tokenized_text[1:-1]] #when using bert token ids, need to minus 1 
        bert_tokens = [re.sub("##", "", i) for i in bert_tokenised_text[1:-1]]
        align = tokenizations.get_alignments(mbert_tokens, bert_tokens)
        bert_ids = [x+1 for x in sorted(list(set(itertools.chain(*[align[0][i-1] for i in mbert_ids]))))]
        return bert_ids
    
    def bert_text_preparation(self, text):
        """
        Preprocesses text input in a way that BERT can interpret.
        """
        tokeniser = BertTokenizer.from_pretrained('bert-base-german-cased')
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokeniser.tokenize(marked_text)
        #print(f"tokenized text: {tokenized_text}\n")
        indexed_tokens = tokeniser.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)
        # convert inputs to tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])
        return tokenized_text, tokens_tensor, segments_tensor

    def mbert_text_preparation(self, text):
        """
        Preprocesses text input in a way that BERT can interpret.
        """
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        #print(f"tokenized text: {tokenized_text}\n")
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)
        # convert inputs to tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])
        return tokenized_text, tokens_tensor, segments_tensor
    
    def get_bert_embeddings_and_ids(self, data, caseless=False):
        """
        Obtains bert embeddings and aligns token ids with mbert 
        """
        bert_model = BertModel.from_pretrained("bert-base-german-cased")
        context_embeddings = []
        progress_bar = tqdm(total=len(data))
        for entry in data:
            if caseless == False:
                entry_sent = entry['sent']
                entry_token_ids = entry['bert_token_ids']
            else:
                entry_sent = entry['caseless_sent']
                entry_token_ids = entry['caseless_bert_ids']
            bert_tokenized_text, bert_tokens_tensor, bert_segments_tensor = self.bert_text_preparation(entry_sent)
            entry_token_ids_bert = self.get_token_ids(entry_sent, entry_token_ids, bert_tokenized_text)
            ind_context_embeddings = {}
            with torch.no_grad():
                # obtain hidden states
                outputs = bert_model(bert_tokens_tensor, bert_segments_tensor, output_hidden_states=True)
                #iterates through layers and embedding layer
                for i in range(len(outputs[2])):
                    token_embedding_specific = []
                    hidden_states = outputs[2][i][0]
                    #loops for specific token index embeddings 
                    for n in entry_token_ids_bert:
                        token_embedding_specific.append(hidden_states[n])
                    #if a single token, appends to dictionary 
                    if len(entry_token_ids_bert) == 1:
                        ind_context_embeddings[i] = token_embedding_specific[0].tolist()
                    #if multiple tokens, gets the tokens and averages them 
                    else:
                        stacked_tensor = torch.stack(token_embedding_specific, dim=0)
                        permuted_stacked_tensor = stacked_tensor.permute(1, 0)
                        ind_context_embeddings[i] = torch.mean(permuted_stacked_tensor, 1).tolist()
            context_embeddings.append(ind_context_embeddings)
            progress_bar.update(1)
        return context_embeddings
    
    def get_bert_embeddings(self, data, caseless=False):
        """
        Obtains bert embeddings 
        """
        context_embeddings = []
        progress_bar = tqdm(total=len(data))
        for entry in data:
            if caseless == False:
                entry_sent = entry['sent']
                entry_token_ids = entry['bert_token_ids']
            else:
                entry_sent = entry['caseless_sent']
                entry_token_ids = entry['caseless_bert_ids']
            tokenized_text, tokens_tensor, segments_tensor = self.mbert_text_preparation(entry_sent)
            ind_context_embeddings = {}
            with torch.no_grad():
                # obtain hidden states
                outputs = self.model(tokens_tensor, segments_tensor, output_hidden_states=True)
                #iterates through layers and embedding layer
                for i in range(len(outputs[2])):
                    token_embedding_specific = []
                    hidden_states = outputs[2][i][0]
                    #loops for specific token index embeddings 
                    for n in entry_token_ids:
                        token_embedding_specific.append(hidden_states[n])
                    #if a single token, appends to dictionary 
                    if len(entry_token_ids) == 1:
                        ind_context_embeddings[i] = token_embedding_specific[0].tolist()
                    #if multiple tokens, gets the tokens and averages them 
                    else:
                        stacked_tensor = torch.stack(token_embedding_specific, dim=0)
                        permuted_stacked_tensor = stacked_tensor.permute(1, 0)
                        ind_context_embeddings[i] = torch.mean(permuted_stacked_tensor, 1).tolist()
            context_embeddings.append(ind_context_embeddings)
            progress_bar.update(1)
        return context_embeddings
    
    def call_train_classifier(self, data_configuration, exp_name, model, caseless=False):
        """
        Gets embeddings, trains classifier per BERT layer, saves results 
        """
        #getting correct data configuration
        if len(data_configuration) == 2:
            case_str = [i for i in data_configuration.keys() if i != "dat"][0]
            training_data = open_json(f"./experiments/{exp_name}/training_data/dat_train.json") + open_json(f"./experiments/{exp_name}/training_data/{case_str}_train.json")
            validation_data = open_json(f"./experiments/{exp_name}/val_data/dat_val.json") + open_json(f"./experiments/{exp_name}/val_data/{case_str}_val.json")
        else:
            reordered_data_config = dict([(v, [k for k, v1 in data_configuration.items() if v1 == v])for v in set(data_configuration.values())])
            small_datasets = [reordered_data_config[x] for x in reordered_data_config if len(reordered_data_config[x]) == 2][0]
            big_dataset = [x for x in data_configuration.keys() if x not in small_datasets][0]
            training_data = open_json(f"./experiments/{exp_name}/training_data/small_{small_datasets[0]}_train.json") + open_json(f"./experiments/{exp_name}/training_data/{big_dataset}_train.json") + open_json(f"./experiments/{exp_name}/training_data/small_{small_datasets[1]}_train.json")
            validation_data =  open_json(f"./experiments/{exp_name}/val_data/acc_val.json") + open_json(f"./experiments/{exp_name}/val_data/dat_val.json") + open_json(f"./experiments/{exp_name}/val_data/nom_val.json")
        test_data = open_json(f"./experiments/{exp_name}/test_data/dat_test.json")
        training_label = [data_configuration[i['case']] for i in training_data]
        test_label = [data_configuration[i['case']] for i in test_data]
        validation_label = [data_configuration[i['case']] for i in validation_data]
        if model == "mbert":
            #get training embeddings
            print("Obtaining Bert embeddings for training data:")
            training_embeddings = self.get_bert_embeddings(training_data, caseless)
            #get validation embeddings
            print("Obtaining Bert embeddings for validation data:")
            validation_embeddings = self.get_bert_embeddings(validation_data, caseless)
            #get test embeddings
            print("Obtaining Bert embeddings for test data:")
            test_embeddings = self.get_bert_embeddings(test_data, caseless)
        elif model == "bert":
            #get training embeddings
            print("Obtaining Bert embeddings for training data:")
            training_embeddings = self.get_bert_embeddings_and_ids(training_data, caseless)
            #get validation embeddings
            print("Obtaining Bert embeddings for validation data:")
            validation_embeddings = self.get_bert_embeddings_and_ids(validation_data, caseless)
            #get test embeddings
            print("Obtaining Bert embeddings for test data:")
            test_embeddings = self.get_bert_embeddings_and_ids(test_data, caseless)
        #training classifier for every layer of BERT
        layer_classifications = {} 
        for i in range(13):
            #train classifier 
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(64,), max_iter=20, random_state=1)
            clf.fit([n[i] for n in training_embeddings], training_label)
            #prediction validation 
            #print("\nPredictions on validation set:")
            val_predict = clf.predict([n[i] for n in validation_embeddings])
            #print(val_predic)
            val_probs = clf.predict_proba([n[i] for n in validation_embeddings])
            #print(val_probs)
            #prediction on test 
            #print("\nPredictions on test set:")
            test_predict = clf.predict([n[i] for n in test_embeddings])
            #print(test_predict)
            test_probs = clf.predict_proba([n[i] for n in test_embeddings])
            #print(clf.predict_proba([n[i] for n in test_embeddings]))
            val_predictions = [{'sent_id': validation_data[n]['sent_id'],
                                'case': validation_data[n]['case'],
                                'actual_classification': validation_label[n],
                                'predicted_classification': int(val_predict[n]), 
                                '0_prob': float(val_probs[n][0]), 
                                '1_prob': float(val_probs[n][1])} for n in range(len(val_predict))]
            test_predictions = [{'sent_id': test_data[n]['sent_id'],
                                'case': test_data[n]['case'],
                                'actual_classification': test_label[n],
                                'predicted_classification': int(test_predict[n]),
                                '0_prob': float(test_probs[n][0]), 
                                '1_prob': float(test_probs[n][1])} for n in range(len(test_predict))]
            layer_classifications[i] = {'val' : val_predictions, 'test': test_predictions}
            self.pickle_classifier(clf, exp_name, data_configuration, i, caseless, model)
        #save 
        os.makedirs(os.path.dirname(f"./experiments/{exp_name}/results/{model}/caseless_{caseless}/{str(data_configuration)}.json"), exist_ok=True)
        save_json(f"./experiments/{exp_name}/results/{model}/caseless_{caseless}/{str(data_configuration)}.json", layer_classifications)
        return
    
    def call_control_train_classifier(self, exp_name, model, caseless=False, case_spef=""):
        #get training and test data
        training_0 = open_json(f"./experiments/control/training_data/{case_spef}0_train.json")
        training_1 = open_json(f"./experiments/control/training_data/{case_spef}1_train.json")
        training_label = [0 for i in training_0] + [1 for i in training_1]
        training_data = training_0 + training_1
        test_data = open_json(f"./experiments/{exp_name}/test_data/dat_test.json")
        test_label = [1 for i in test_data]
        if model == "mbert":
            #get training embeddings
            print("Obtaining Bert embeddings for training data:")
            training_embeddings = self.get_bert_embeddings(training_data, caseless)
            #get test embeddings
            print("Obtaining Bert embeddings for test data:")
            test_embeddings = self.get_bert_embeddings(test_data, caseless)
        elif model == "bert":
            #get training embeddings
            print("Obtaining Bert embeddings for training data:")
            training_embeddings = self.get_bert_embeddings_and_ids(training_data, caseless)
            #get test embeddings
            print("Obtaining Bert embeddings for test data:")
            test_embeddings = self.get_bert_embeddings_and_ids(test_data, caseless)
        #training classifier for every layer of BERT
        layer_classifications = {} 
        for i in range(13):
            #train classifier 
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(64,), max_iter=20, random_state=1)
            clf.fit([n[i] for n in training_embeddings], training_label)
            #prediction on test 
            #print("\nPredictions on test set:")
            test_predict = clf.predict([n[i] for n in test_embeddings])
            #print(test_predict)
            test_probs = clf.predict_proba([n[i] for n in test_embeddings])
            #print(clf.predict_proba([n[i] for n in test_embeddings]))
            test_predictions = [{'sent_id': test_data[n]['sent_id'],
                                'case': test_data[n]['case'],
                                'actual_classification': test_label[n],
                                'predicted_classification': int(test_predict[n]),
                                '0_prob': float(test_probs[n][0]), 
                                '1_prob': float(test_probs[n][1])} for n in range(len(test_predict))]
            layer_classifications[i] = {'test': test_predictions}
            self.pickle_classifier(clf, exp_name, f"{case_spef}random", i, caseless, model)
        #save results
        os.makedirs(os.path.dirname(f"./experiments/{exp_name}/results/{model}/caseless_{caseless}/{case_spef}random.json"), exist_ok=True)
        save_json(f"./experiments/{exp_name}/results/{model}/caseless_{caseless}/{case_spef}random.json", layer_classifications)
        return

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("model", type=str, help="The model you want to test: bert or mbert.")
    argument_parser.add_argument("control", type=str, help="If you want the control probe (as detailed in appendix F), put True, otherwise False")
    argument_parser.add_argument("dataset_type", type=str, help="The specific dataset charateristics you want: random, noun_only, pronoun_only or same_token_number")
    argument_parser.add_argument("case_configuration", type=int, help="The specific case configuration you want, indicated with an int: 1 = {'acc':0, 'dat':1}, 2 = {'nom':0, 'dat':1}, 3 = {'nom':0, 'acc':0, 'dat':1}, 4 = {'nom':1, 'acc':0, 'dat':1}, or 5 = {'nom':0, 'acc':1, 'dat':1}")
    argument_parser.add_argument("caseless", type=str, help="If you want to run the cased experiment put False, if you want the caseless experiment put True")
    argument_parser.add_argument("--case_spef", type=str, help="If control is True, put the specific control probe (as detailed in appendix F): nom, acc, or all")
    arguments = argument_parser.parse_args()

    #raises error if model is wrong 
    if arguments.model == "bert":
        model = "bert"
    elif arguments.model == "mbert":
        model = "mbert"
    else:
        raise TypeError("model has to be either bert or mbert.")

    #assigns case config and raises error if wrong int inputted
    if arguments.case_configuration == 1:
        case_config = {'acc':0, 'dat':1}
    elif arguments.case_configuration == 2:
        case_config = {'nom':0, 'dat':1}
    elif arguments.case_configuration == 3:
        case_config = {'nom':0, 'acc':0, 'dat':1}
    elif arguments.case_configuration == 4:
        case_config = {'nom':1, 'acc':0, 'dat':1}
    elif arguments.case_configuration == 5:
        case_config = {'nom':0, 'acc':1, 'dat':1}
    else:
        raise TypeError("Wrong int has been input. You can only have 1, 2, 3, 4, or 5. See the --help message for which case configuration takes which int. If control is True, put 1.")

    #raises error if control = True but there is no case_spef or wrong case_spef
    if arguments.control == "True":
        if arguments.case_spef == "nom":
            case_spef =  "nom_"
        elif arguments.case_spef == "acc":
            case_spef =  "acc_"
        elif arguments.case_spef == "all":
            case_spef =  ""   
        else:
            raise TypeError("case_spef cannot be left empty whilst control is True. Please input either acc, nom, or all.")
   
    #assigns dataset_type otherwise raises error 
    if arguments.dataset_type == "random":
        dataset_type = "random"
    elif arguments.dataset_type == "noun_only":
        dataset_type = "noun_only"
    elif arguments.dataset_type == "pronoun_only":
        dataset_type = "pronoun_only"
    elif arguments.dataset_type == "same_token_number":
        dataset_type = "same_token_number"
    else:
        raise TypeError("dataset_type has to be random, noun_only, pronoun_only, or same_token_number. If control is True, put random")

    #assigns caseless otherwise raises error 
    if arguments.caseless == "True":
        caseless = True 
    elif arguments.caseless == "False":
        caseless = False 
    else:
        raise TypeError("caseless has to be either True or False")
    
    #checks control otherwise raises error
    if arguments.control == "True":
        control = True 
    elif arguments.control == "False":
        control = False 
    else: 
        raise TypeError("control has to be either True or False")


    train_classifier = TrainClassifier(BertTokenizer.from_pretrained('bert-base-multilingual-cased'), 
                                      BertModel.from_pretrained("bert-base-multilingual-cased"))
    
    if control == False:
        train_classifier.call_train_classifier(case_config, dataset_type, model, caseless=caseless)
    
    if control == True:
        train_classifier.call_control_train_classifier("control", model, caseless=caseless, case_spef=case_spef)

