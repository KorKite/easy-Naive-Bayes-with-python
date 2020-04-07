training_data = [[["Click", "win", "prize"], "spam"], [["Click", "meeting", "setup", "meeting"], "not"],
             [["Prize","free","prize"], "spam"], [["Prize", "free", "prize"], "spam"]]

test_case = ["Free", "setup", "meeting", "free"]

import numpy as np


class binary_Naive_bayes:

    def __init__(self, train_data):
        # train_data 를 받음
        self.td = np.array(train_data)
        self.one = []
        self.two = []
        self.one_count = 0
        self.two_count = 0
        self.tag_name1 = None
        self.tag_name2 = None
        first = True

        # data tag에 따라 두개의 리스트에 분리되도록 설정를 / 테그의 이름도 설정
        for data, tag in self.td:
            if first:
                self.tag_name1 = tag
                self.one = self.one + [w.lower() for w in data]
                self.one_count += 1
                first = False
            else:
                if self.tag_name1 == tag:
                    self.one = self.one + [w.lower() for w in data]
                    self.one_count += 1
                else:
                    self.tag_name2 = tag
                    self.two = self.two + [w.lower() for w in data]
                    self.two_count += 1

        self.one = np.array(self.one)
        self.two = np.array(self.two)

    def normal(self, test_data):
        # without smoothing, zero can be occur
        cls_one = len(self.one)
        cls_two = len(self.two)
        test_data = np.array(test_data)
        # 계산식
        fin_one = 1
        fin_two = 1
        for test in test_data:
            fin_one *= len(np.where(test == self.one)) / cls_one
        fin_one = fin_one * (self.one_count / (self.one_count + self.two_count))
        for test in test_data:
            fin_two *= len(np.where(test == self.two))/ cls_two
        fin_two = fin_two * (self.two_count / (self.one_count + self.two_count))
        if fin_two > fin_one:
            return self.tag_name2
        else:
            return self.tag_name1

    def laplace_smoothing(self, test_data):
        # with laplace smoothing, prevent zero
        cls_one = len(self.one)  # 1번째 클래스의 길이
        cls_two = len(self.two)  # 2번째 클래스의 길이
        la_first = [w.lower() for w in self.one]
        la_second = [w.lower() for w in self.two]
        la_total = set(la_first + la_second)  # 모든 단어의 종류
        la_total = np.array((la_total))
        # 계산식 (조건부확률)
        fin_one = 1
        fin_two = 1
        for test in test_data:
            fin_one *= (len(np.where(test.lower() == self.one)) + 1) / (cls_one + len(la_total))
        fin_one = fin_one * (self.one_count / (self.one_count + self.two_count))
        for test in test_data:
            fin_two *= (len(np.where(test.lower() == self.two)) + 1) / (cls_two + len(la_total))
        fin_two = fin_two * (self.two_count / (self.one_count + self.two_count))

        print("{} - {} /// {} - {}".format(self.tag_name1, fin_one, self.tag_name2, fin_two))
        if fin_two > fin_one:
            return self.tag_name2
        else:
            return self.tag_name1

    def use_log(self, test_data):
        # with laplace smoothing, prevent zero plus use log when calculate
        cls_one = len(self.one)  # 1번째 클래스의 길이
        cls_two = len(self.two)  # 2번째 클래스의 길이
        la_first = [w.lower() for w in self.one]
        la_second = [w.lower() for w in self.two]
        la_total = set(la_first + la_second)  # 모든 단어의 종류

        import math
        # 계산식 (조건부확률)
        fin_one = 0
        fin_two = 0
        for test in test_data:
            fin_one += math.log10((self.one.count(test.lower()) + 1) / (cls_one + len(la_total)))
        fin_one = math.exp((fin_one) * (self.one_count / (self.one_count + self.two_count)))
        for test in test_data:
            fin_two += math.log10((self.two.count(test.lower()) + 1) / (cls_two + len(la_total)))
        fin_two = math.exp((fin_two) * (self.two_count / (self.one_count + self.two_count)))

        print("{} - {} /// {} - {}".format(self.tag_name1, fin_one, self.tag_name2, fin_two))
        if fin_two > fin_one:
            return self.tag_name2
        else:
            return self.tag_name1


bnb = binary_Naive_bayes(training_data)
print(bnb.normal(test_case))
print()
print(bnb.laplace_smoothing(test_case))
print()
print(bnb.use_log(test_case))
