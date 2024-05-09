import numpy as np
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, plot_model
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
import itertools
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D, BatchNormalization
import matplotlib.pyplot as plt
from data_process_MCNN import x_all, y_all
import csv
X = x_all
Y = y_all
print(X.shape)
print(Y.shape)
Y_onehot = np_utils.to_categorical(Y)
print(Y_onehot.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0)
def baseline_model():
    model = Sequential()
    model.add(Convolution1D(filters=3, kernel_size=8, strides=11, input_shape=(500, 1), padding="same"))

    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, padding='same'))

    model.add(Convolution1D(filters=3, kernel_size=8, strides=11, input_shape=(250, 1), padding="same"))
    model.add(
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    plot_model(model, to_file='./result/model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=10, verbose=1)
history = estimator.fit(X_train, Y_train, validation_data=(X_test,Y_test))
# Save epochs and loss to CSV
with open('./result/loss.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Loss','Val_Loss'])
    for epoch, (loss,val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
        writer.writerow([epoch + 1, loss, val_loss])
# Save epochs and accuracy to CSV
with open('./result/accuracy.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Accuracy', "Val_Accuracy"])
    for epoch, (acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
        writer.writerow([epoch + 1, acc,val_acc])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(history.history['loss'], label=u'train_loss')
plt.plot(history.history['val_loss'], label=u'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.show()
plt.plot(history.history['accuracy'], label=u'train_acc')
plt.plot(history.history['val_accuracy'], label=u'val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

def visual(model, data, num_layer=1):
    layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
    f1 = layer([data])[0]
    print(f1.shape)
    num = f1.shape[-1]
    print(num)
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(int(np.ceil(np.sqrt(num))), int(np.ceil(np.sqrt(num))), i + 1)
        plt.imshow(f1[:, :, i] * 255, cmap='gray')
        plt.axis('off')
    plt.show()
def calculate_metrics(conf_mat):
    TP = np.diag(conf_mat)
    FP = np.sum(conf_mat, axis=0) - TP
    FN = np.sum(conf_mat, axis=1) - TP
    TN = np.sum(conf_mat) - (FP + FN + TP)
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    F1 = 2*precision*recall/(precision+recall)
    return accuracy,precision, recall, specificity,F1
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Show percentage：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('Show specific numbers：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, ('phenylalanine',"methionine","lysine",'leucine','threonine','aspartame','fructose','glucose',"lactose monohydrate","sucrose"), rotation=45)
    plt.yticks(tick_marks, ('phenylalanine',"methionine","lysine",'leucine','threonine','aspartame','fructose','glucose',"lactose monohydrate","sucrose"))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('predicted category')
    plt.xlabel('real category')
    plt.savefig('./result/suan_tang.svg', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()
def plot_confuse(model, x_val, y_val):
    predictions_temp = model.predict(x_val)
    predictions = np.argmax(predictions_temp, axis=1)
    truelabel = y_val.argmax(axis=-1)
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.tight_layout()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))

    accuracy,precision, recall, specificity,F1 = calculate_metrics(conf_mat)
    print("accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F1:", F1)
model_json = estimator.model.to_json()
with open('./result/model', 'w') as json_file:
    json_file.write(model_json)
estimator.model.save_weights('./result/model.json.h5')
estimator.model.save(
'./result/model_with_weights.h5'
)
json_file = open(r'./result/model')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('./result/model.json.h5')
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
predicted = loaded_model.predict(X)
predicted_temp = loaded_model.predict(X)
predicted_label=np.argmax(predicted_temp,axis=1)
print("predicted label:\n " + str(predicted_label))
plot_confuse(estimator.model, X_test, Y_test)





