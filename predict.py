import os, cv2, torch
import numpy as np

if __name__ == '__main__':
    with open('classes.txt') as f:
        label = list(map(lambda x:x.strip(), f.readlines()))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model = torch.load('model.pht').to(DEVICE)

    while True:
        img_path = input('Imput Image Path:')
        try:
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            img = np.transpose(img, axes=[2, 0, 1]) / 255.0
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img).to(DEVICE).float()
            pred = np.argmax(model(img).cpu().detach().numpy()[0])
            print('Image Path:{}, Pred Class:{}'.format(img_path, label[pred]))
        except:
            print('Error, Try Again!')
