from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import AUC
import pandas as pd
import numpy as np
import tensorflow as tf 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from twin_preprocessing import twin_preprocessing, merge_into_one, plot_loss, plot_accuracy, plot_rmse

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# Creating the PropensityNet model
def create_propensity_net(input_shape, output_units=1, hidden_layers=[32, 64, 128,64], dropout_rate=0.3, l2_reg=0.0001):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    # Adding hidden layers
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.BatchNormalization())
    
    # Output layer
    model.add(layers.Dense(output_units))

    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=[rmse, 'mae'])#, AUC(name='auc')])
    return model

merged_data = pd.read_csv('Harvard/MIT67900/proj/price_final.csv')
# hidden_layers=[47, 1175, 293, 587, 1175, 376, 94, 188]
# Split the features and labels from the dataset
# X_features = merged_data.drop(['T', 'Outcome'], axis=1)  # Remove the label columns to isolate the features
scaler = StandardScaler()
X = scaler.fit_transform(merged_data.drop(['noise_price', 'noise_demand', 'mu0','mut', 'structural', 'outcome', 'price', 'price_category'], axis=1))

# Encode the target variable
category_mapping = {'low': 0, 'medium': 1, 'high': 2}
y = merged_data['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
input_shape = X_train.shape[1:]
propensity_model = create_propensity_net(input_shape)
# Early stopping callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
# Define a learning rate decay function
def lr_schedule(epoch, lr):
    if epoch > 10:
        lr = lr * 0.1
    return lr

# Create a LearningRateScheduler callback
lr_scheduler = callbacks.LearningRateScheduler(lr_schedule, verbose=1)

# Train the model
history = propensity_model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=128, 
    verbose=1, 
    validation_split=0.1, 
    callbacks=[early_stopping, reduce_lr]
)  # Reduced epochs for quicker execution

# Evaluate the model on the test set
evaluation = propensity_model.evaluate(X_test, y_test, verbose=1)

evaluation

plot_rmse(history)
plot_loss(history)

propensity_scores = propensity_model.predict(X).ravel()
# Calculating weights
# This is a simplified example; the actual implementation may vary based on the specifics of your study
weights = 1 / propensity_scores
weights[np.isinf(weights)] = np.max(weights[~np.isinf(weights)])
weights[np.isnan(weights)] = np.mean(weights[~np.isnan(weights)])

# Calculate the weighted average outcome
weighted_outcome = np.sum(merged_data['outcome'] * weights) / np.sum(weights)

# Calculate ATE for a unit increase in price
# This is a simplified approach; a more rigorous method would involve regression modeling
merged_data['weighted_outcome'] = merged_data['outcome'] * weights
lm = LinearRegression()
lm.fit(merged_data[['price']], merged_data['weighted_outcome'])
ate = lm.coef_[0]
print(f'Estimated ATE of a unit increase in price: {ate}')




