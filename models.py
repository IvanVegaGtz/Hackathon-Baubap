import numpy as np
import lightgbm as lgb
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import backend as K
#from tensorflow.keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.ensemble import RandomForestClassifier

def get_rf(X_train, y_train):
    X_train = X_train.fillna(0)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def predict_rf(rf_model, X_test):
    X_test = X_test.fillna(0)
    probs = rf_model.predict_proba(X_test)[:, 1]
    return probs


def get_lgbm(X_train, y_train, X_test, y_test):
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    test_data_lgb = lgb.Dataset(X_test, label=y_test)

    # Parametros
    params_k = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'subsample': 0.95,
                'learning_rate': 0.01,
                'max_depth': 5,
                'num_leaves': 100,
                'feature_fraction': 0.9,
                'is_unbalance': True,
                "random_seed":42,
                'verbose': -1
    }

    # Entrenamiento
    model = lgb.train(params_k,
                        train_data_lgb,
                        valid_sets=[test_data_lgb, train_data_lgb],
                        num_boost_round=1000
                        #early_stopping_rounds=100, verbose_eval=50
                    )

    #ax = lgb.plot_importance(model, max_num_features=20)
    #plt.show()
    return model

def predict_lgbm(model_lgbm, X_test):
     probs = model_lgbm.predict(X_test, num_iteration=model_lgbm.best_iteration)
     return probs


def get_svm(X_train, y_train):
     X_train = X_train.fillna(0)
     svm_classifier = SVC(kernel='linear', probability = True, class_weight='balanced', random_state=42)
     svm_classifier.fit(X_train, y_train)
     return svm_classifier

def predict_svm(svm_classifier, X_test):
     X_test = X_test.fillna(0)
     probs = svm_classifier.predict_proba(X_test)
     probs = probs[:,1]
     return probs


def brier_score(y_true, y_pred):
    # Asumiendo que y_pred es la probabilidad de la clase positiva
    return K.mean(K.square(y_true - y_pred))

def get_nn(X_train, y_train, X_test, y_test, epochs, batch_size):

    # Se reemplazan valores nulos
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    # Construimos la arquitectura de la red neuronal
    '''
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    '''
    # Capa de entrada
    input_layer = Input(shape=(X_train.shape[1],))
    # Capa oculta con regularización L2
    hidden_layer1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    hidden_layer2 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(hidden_layer1)
    # Capa de salida
    output_layer = Dense(1, activation='sigmoid')(hidden_layer2)
    # Construir el modelo
    model = Model(inputs=input_layer, outputs=output_layer)

    #Compilamos el modelo
    model.compile(loss=brier_score, optimizer='adam', metrics=['accuracy'])
    # Calculamos los pesos de clase para abordar el desbalance
    class_weights = compute_class_weight(class_weight ='balanced', classes = np.unique(y_train), y = y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    # Antes de entrenar el modelo, se convierten los datos de DataFrame a tensores de TensorFlow
    X_train_tensor = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
    X_val_tensor = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
    y_val_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.float32)
    # Definir ModelCheckpoint para guardar el mejor modelo
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    # Entrenamiento del modelo
    model.fit(X_train_tensor, 
              y_train_tensor, 
              epochs=epochs, 
              batch_size= batch_size, 
              class_weight=class_weight_dict, 
              validation_data=(X_val_tensor, y_val_tensor),
              callbacks=[checkpoint])
    return model

def predict_nn(model, X_test):
    X_test = X_test.fillna(0)
    probs = model.predict(X_test)
    probs = probs.reshape(-1)
    return probs









