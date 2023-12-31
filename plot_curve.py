import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.log')
acc = data['acc']
loss = data['loss']
val_acc = data['val_acc']
val_loss = data['val_loss']

plt.figure(figsize=(8, 10))

plt.subplot(2, 1, 1)
plt.plot(loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.title('loss')

plt.subplot(2, 1, 2)
plt.plot(acc, label='train_acc')
plt.plot(val_acc, label='val_acc')
plt.legend()
plt.title('acc')

plt.show()