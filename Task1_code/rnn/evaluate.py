from utils import *



def _evaluate(model, line_tensor):
    hidden = model.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output

def evaluate(test_data, model, all_categories):
    # Go through held-out examples and record which are correctly guessed
    total = 0
    right = 0
    for category in all_categories:
        for line in test_data[category]:
            _, line, category_tensor, line_tensor = parseTestExample(all_categories, category, line)
            output = _evaluate(model, line_tensor, )
            guess, guess_i = categoryFromOutput(output, all_categories)
            category_i = all_categories.index(category)
            if guess_i == category_i:
                right += 1
            total += 1
    print('accuracy:', right / total)
    return right / total


def cnn_evaluate(test_data, model, all_categories):
    # Go through held-out examples and record which are correctly guessed
    total = 0
    right = 0
    for category in all_categories:
        for line in test_data[category]:
            _, line, category_tensor, line_tensor = parseTestExample(all_categories, category, line)
            line_tensor = line_tensor.transpose(0, 1)
            output = model(line_tensor)
            guess, guess_i = categoryFromOutput(output, all_categories)
            category_i = all_categories.index(category)
            if guess_i == category_i:
                right += 1
            total += 1
    print('accuracy:', right / total)
    return right / total

    # category, line, category_tensor, line_tensor = randomTrainingExample()
    # output = evaluate(line_tensor)
    # guess, guess_i = categoryFromOutput(output)
    # category_i = all_categories.index(category)
    # confusion[category_i][guess_i] += 1
        