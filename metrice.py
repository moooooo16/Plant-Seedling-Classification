import os, cv2, torch, itertools, tqdm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, name='test'):
    plt.figure(figsize=(10, 10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(name + title, fontsize=11)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90, fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j], 2), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=9)
    plt.xlabel('Predicted label', fontsize=9)
    plt.savefig("confusion_matrix.png", dpi=150)
    return cm


if __name__ == '__main__':
    with open('label.txt') as f:
        label = list(map(lambda x:x.strip(), f.readlines()))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model = torch.load('model.pht').to(DEVICE)

    y_pred, y_true = [], []
    for i in tqdm.tqdm(os.listdir('test')):
        base_path = os.path.join('test', i)
        for j in os.listdir(base_path):
            img = cv2.imread(os.path.join(base_path, j))
            img = cv2.resize(img, (224, 224))
            img = np.transpose(img, axes=[2, 0, 1]) / 255.0
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img).to(DEVICE).float()
            pred = np.argmax(model(img).cpu().detach().numpy()[0])
            y_pred.append(pred)
            y_true.append(int(i))

    y_pred, y_true = np.array(y_pred), np.array(y_true)
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, label)
