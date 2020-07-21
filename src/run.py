"""
Written by Ehsan
"""
import pandas as pd
import torch
from utils import DataSet
from config import *
import pickle as pkl
from torchtext import data
import numpy as np

# def evaluate(model, iterator):
#     """
#     method for evaluate model
#     :param model: your creation model
#     :param iterator: your iterator
#     :param criterion: your criterion
#     :return: loss, accuracy, precision, recall and F1 score for each epoch
#     """
#     model.eval()
#
#     with torch.no_grad():
#         for batch in iterator:
#             predictions = model(batch.question, batch.answer)
#             print(predictions)
#
#
# def count_parameters(input_model):
#     """
#     method for calculate number of model's parameter
#     :param input_model: model
#     :return: number of parameters
#     """
#
#     return sum(p.numel() for p in input_model.parameters() if p.requires_grad)


class question_matching:
    def __init__(self, **kwargs):
        self.test_csv_path = kwargs["test_csv_path"]
        self.knowledge_csv_path = kwargs["knowledge_csv_path"]
        self.model_path = kwargs["model_path"]
        self.result_path = kwargs["result_path"]
        self.run()

    def run(self):
        test_question = self.read_test_csv(self.test_csv_path)
        knowledge_q, knowledge_a = self.read_knowledge_data(self.knowledge_csv_path)
        model = self.load_model(self.model_path)
        df = pd.DataFrame()
        i = 0
        a = True
        for query in test_question:
            i += 1
            l = self.evaluate(model, self.create_knowledge_base_iterator(query, knowledge_q))
            df["query"] = [query, query, query]
            df["question"] = [knowledge_q[l[0]], knowledge_q[l[1]], knowledge_q[l[2]]]
            df["answer"] = [knowledge_a[l[0]], knowledge_a[l[1]], knowledge_a[l[2]]]
            if a:
                df.to_csv(self.result_path, index=False)
                a = False
            else:
                df.to_csv(self.result_path, index=False, mode="a", header=False)
            print(f"{i} question pass.")

    @staticmethod
    def read_test_csv(input_path):
        input_df = pd.read_csv(input_path)
        # print("load test data")
        return input_df.test_questions

    @staticmethod
    def read_knowledge_data(input_path):
        input_df = pd.read_csv(input_path)
        input_df = input_df.astype({"NormalQuestionContent": "str"})
        input_df = input_df.astype({"NormalAnswerContent": "str"})
        # print("load knowledge base")
        return input_df.NormalQuestionContent, input_df.NormalAnswerContent

    @staticmethod
    def load_model(input_path):
        # print("load model")
        model = torch.load(TEST_MODEL_PATH, map_location=DEVICE)
        return model

    @staticmethod
    def create_knowledge_base_iterator(query, input_questions):
        with open(TEXT_FIELD_PATH, "rb") as f:
            text_field = pkl.load(f)
        query_data = []
        question_data = []

        for question in input_questions:
            query_data.append(query)
            question_data.append(question)
        df = pd.DataFrame({"query": query_data, "questions": question_data})

        datafields = [('query', text_field), ('questions', text_field)]
        test_examples = [data.Example.fromlist(i, datafields) for i in
                         df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)

        test_iterator = data.BucketIterator(
            test_data,
            batch_size=BATCH_SIZE,
            sort=False,
            device=DEVICE,
            shuffle=False)
        # print("create iterator")
        return test_iterator

    @staticmethod
    def evaluate(model, iterator):
        # print("start evaluating")
        model.eval()
        i=0
        my_list = []
        with torch.no_grad():
            for batch in iterator:
                predictions = model(batch.questions, batch.query)
                for p in predictions[:, 1]:
                    my_list.append(p.float())
        my_list = np.array(my_list)
        print(my_list.argsort()[-3:][::-1])
        return my_list.argsort()[-3:][::-1]


if __name__ == "__main__":
    mc = question_matching(test_csv_path=TEST_PATH, knowledge_csv_path=KNOWLEDGE_PATH,
                           model_path=TEST_MODEL_PATH, result_path=MODEL_RESULT_PATH)


