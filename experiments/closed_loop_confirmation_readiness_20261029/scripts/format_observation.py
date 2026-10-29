"""Label-free formatting for conditional future confirmation, not an evaluation runner."""


def arc_easy(question, choice_labels, choice_texts):
    assert isinstance(question, str)
    assert len(choice_labels) == len(choice_texts) >= 2
    assert len(set(choice_labels)) == len(choice_labels)
    assert all(isinstance(t, str) and len(t) > 0 for t in choice_texts)
    prompt = 'Question: ' + question + '\nAnswer:'
    return {'task': 'arc_easy', 'question': question,
            'variants': {'official': [[prompt, ' ' + text] for text in choice_texts]},
            'choice_labels': list(choice_labels),
            'choice_char_lengths': [len(text) for text in choice_texts],
            'primary_metric': 'acc_norm', 'secondary_metric': 'acc'}


def commonsense_qa(question, choice_labels, choice_texts):
    assert isinstance(question, str)
    assert list(choice_labels) == list('ABCDE') and len(choice_texts) == 5
    assert all(isinstance(t, str) for t in choice_texts)
    question = question.strip()
    prompt = 'Question: ' + question + '\n' + '\n'.join(label + '. ' + text for label, text in zip(choice_labels, choice_texts)) + '\nAnswer:'
    return {'task': 'commonsense_qa', 'question': question,
            'variants': {'official': [[prompt, ' ' + label] for label in choice_labels]},
            'choice_labels': list(choice_labels), 'choice_char_lengths': [1] * 5,
            'primary_metric': 'acc', 'secondary_metric': None}
