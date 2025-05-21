import jinja2
from typing import List

from .dataset import DatasetMeta


class Decoder:
    # functions that transform the result to another str
    transforms: list[callable] = []

    def decode(self, _answer: str) -> list[str]:
        raise NotImplementedError()

    def apply(self, transform: callable) -> "Decoder":
        self.transforms.append(transform)
        return self

    def lower_cased(self) -> "Decoder":
        return self.apply(lambda x: x.lower())

    def remove_trailing(self, char_list: list[str]) -> "Decoder":
        def remove_trailing(x: str) -> str:
            while len(x) > 0 and x[-1] in char_list:
                x = x[:-1]
            return x

        self.apply(remove_trailing)
        return self

    def remove_tokens(self, char_list: list[str]) -> "Decoder":
        for char in char_list:
            self.apply(lambda x: x.replace(char, ""))
        return self


class SeperatorDecoder(Decoder):
    def __init__(self, seperator: str, class_names: list[str]) -> None:
        self.seperator = seperator
        self.class_names = class_names

    def decode(self, answer: str) -> list[str]:
        for transform in self.transforms:
            answer = transform(answer)
        results = []
        for component in answer.split(self.seperator):
            for class_name in self.class_names:
                if component.endswith(class_name):
                    results.append(class_name)
                    break
        return results


class Serializer:
    def __init__(self, meta: DatasetMeta) -> None:
        self.meta = meta

    def serialize(self, x, y) -> str:
        raise NotImplementedError()

    def format_desc(self):
        return None

    def answer_requirement(self, ncases: int) -> str:
        return None
        
    def unknown_correction_example(self) -> str:
        
        label_names = [label.name for label in self.meta.labels]
        
        example = f"""
Incorrect output for 4 lines:
{label_names[0]}
{label_names[0]}
unknown 
{label_names[-1]}

Since 'unknown' is not allowed, you should use your knowledge to make predictions for the 'unknown' ones.

One possible corrected output:
{label_names[0]}
{label_names[0]} 
{label_names[0]}
{label_names[-1]}

Output ONLY the corrected predictions(no 'unknown'):"""
        return example

    @property
    def answer_decoder(self) -> Decoder:
        return None


class TabularSerializer(Serializer):
    def __init__(self, meta: DatasetMeta) -> None:
        super().__init__(meta)

    def _gen_feat_desc(self, x) -> str:
        result = ""
        for i in range(len(x)):
            feat_repr = self.meta.value_repr(i, x[i])
            result += f"{feat_repr}, "

        return result[:-2]

    def serialize(self, x, y) -> str:
        if y is not None:
            label_text = self.meta.find_label(y).name
        else:
            label_text = "<RESULT>".format(self.meta.label_meaning)
        return self._gen_feat_desc(x) + " " + label_text

    def format_desc(self):
        desc = "Here's how the data will be presented. For each line:\n"
        for feat in self.meta.features:
            desc += f"<{feat.name}>, "
        desc = desc[:-2] + " <RESULT>".format(self.meta.label_meaning)
        return desc

    def answer_requirement(self, ncases: int) -> str:
        req = 'Please make prediction on the following {} lines containing "<RESULT>", filling "<RESULT>" with one of '.format(
            ncases
        )
        for label in self.meta.labels:
            req += f' "{label.name}",'
        req += " and reply each prediction in a new line. Make sure there are exactly {} lines.".format(
            ncases
        )
        
        return req

    @property
    def answer_decoder(self) -> Decoder:
        # just assume the class names are in lower cases
        return (
            SeperatorDecoder("\n", [x.name for x in self.meta.labels])
            .lower_cased()
            .remove_tokens(["<", ">", " "])
            .remove_trailing([",", ".", '"', "'"])
        )


class ListSerializer(Serializer):
    def serialize(self, x, y) -> str:
        ret = ""
        for i in range(len(x)):
            feat_name = self.meta.features[i].name
            feat_repr = self.meta.value_repr(i, x[i])
            ret += f"{feat_name}: {feat_repr}; "
        if y is not None:
            label = self.meta.find_label(y)
            label_text = "The {}: {}".format(self.meta.label_meaning, label.name)
        else:
            label_text = "The {}: <RESULT>".format(self.meta.label_meaning)

        ret += f"{label_text}"
        return ret

    def answer_requirement(self, ncases: int) -> str:
        req = 'Please make prediction on the following {} lines containing "<RESULT>", filling "<RESULT>" with one of '.format(
            ncases
        )
        for label in self.meta.labels:
            req += f' "{label.name}",'
            
        req += " and reply each prediction in a new line with " + " or ".join(
            ['"' + x.name + '"' for x in self.meta.labels]
        )

        req += ". Make sure there are exactly {} lines.".format(
            ncases
        )
        
        return req

    @property
    def answer_decoder(self) -> Decoder:
        return (
            SeperatorDecoder("\n", [x.name for x in self.meta.labels])
            .lower_cased()
            .remove_tokens(["<", ">", " "])
            .remove_trailing([",", "."])
        )


class TextSerializer(Serializer):
    def serialize(self, x, y) -> str:
        ret = ""
        for i in range(len(x)):
            feat_name = self.meta.features[i].name
            feat_repr = self.meta.value_repr(i, x[i])
            ret += f"The {feat_name} is {feat_repr}. "

        if y is not None:
            label = self.meta.find_label(y)
            label_text = "The {} is: {}".format(self.meta.label_meaning, label.name)
        else:
            label_text = "The {} is: <RESULT>".format(self.meta.label_meaning)

        ret += f"{label_text}"
        return ret

    def answer_requirement(self, ncases: int) -> str:
        req = 'Please make prediction on the following {} lines containing "<RESULT>", filling "<RESULT>" with one of '.format(
            ncases
        )
        for label in self.meta.labels:
            req += f' "{label.name}",'
        req += " and reply each prediction in a new line. Make sure there are exactly {} lines.".format(
            ncases
        )
        
        return req

    @property
    def answer_decoder(self) -> Decoder:
        return (
            SeperatorDecoder("\n", [x.name for x in self.meta.labels])
            .lower_cased()
            .remove_tokens(["<", ">", " "])
            .remove_trailing([",", "."])
        )


class CustomDecoder:
    def decode(self, text):
     
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        return lines


def gen_prompt(
    master_template: jinja2.Template,
    row_serializer: Serializer,
    x_train,
    y_train,
    x_test,
    extra_rules: list[str],
    num_tests_per_round: int,
) -> tuple[list[str], list[tuple[int]]]:
    prompts: list[str] = []

    # Generate string representation of examples with row_serializer
    x_train_str = []
    for i in range(len(x_train)):
        x_train_str.append(row_serializer.serialize(x_train[i], y_train[i]))

    # Generate string representation of test cases
    x_test_str = []
    for i in range(len(x_test)):
        x_test_str.append(row_serializer.serialize(x_test[i], None))

    format_desc = row_serializer.format_desc()

    total_tests = x_test.shape[0]
    test_splits = []
    current = 0

    while current < total_tests:
        if num_tests_per_round > 0:
            num_cases = min(num_tests_per_round, total_tests - current)
        else:
            num_cases = total_tests

        # generate from template
        prompt = master_template.render(
            meta=row_serializer.meta,
            examples=x_train_str,
            rules=extra_rules,
            format_desc=format_desc,
            prediction_intro=row_serializer.answer_requirement(num_cases),
            tests=x_test_str[current : current + num_cases],
            unknown_correction_example=row_serializer.unknown_correction_example()
        )
        prompt = prompt.strip() + "\n"

        prompts.append(prompt)
        test_splits.append((current, current + num_cases))
        current += num_cases

    return prompts, test_splits
