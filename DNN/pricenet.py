from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import AUC
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from twin_preprocessing import twin_preprocessing, merge_into_one, plot_loss, plot_accuracy, confusion3d



# Creating the PropensityNet model
def create_propensity_net(input_shape, output_units=3, hidden_layers=[32, 64], dropout_rate=0.1, l2_reg=0.0001):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    # Adding hidden layers
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.BatchNormalization())
    
    # Output layer
    model.add(layers.Dense(output_units, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])#, AUC(name='auc')])
    return model

merged_data = pd.read_csv('Harvard/MIT67900/proj/price_final.csv')
# hidden_layers=[47, 1175, 293, 587, 1175, 376, 94, 188]
# Split the features and labels from the dataset
# X_features = merged_data.drop(['T', 'Outcome'], axis=1)  # Remove the label columns to isolate the features
scaler = StandardScaler()
X = scaler.fit_transform(merged_data.drop(['noise_price', 'noise_demand', 'mu0','mut', 'structural', 'outcome', 'price', 'price_category'], axis=1))

# Encode the target variable
category_mapping = {'low': 0, 'medium': 1, 'high': 2}
y = merged_data['price_category'].map(category_mapping)


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

propensity_scores = propensity_model.predict(X)
print(slice(propensity_scores))
# Add the propensity scores to your dataset
weights = np.zeros_like(y)

for i, category in enumerate(['low', 'medium', 'high']):
    treatment_mask = (y == i)
    weights[treatment_mask] = 1 / propensity_scores[treatment_mask, i]

# Replace infinities or NaNs in weights with some large number or mean of the weights
weights[np.isinf(weights)] = np.max(weights[~np.isinf(weights)])
weights[np.isnan(weights)] = np.mean(weights[~np.isnan(weights)])

# ATE
weighted_outcomes = merged_data['outcome'] * weights
ate_values = {}
total = 0
for i, category in enumerate(['low', 'medium', 'high']):
    treatment_mask = (y == i)
    weighted_avg_outcome = np.sum(weighted_outcomes[treatment_mask]) / np.sum(weights[treatment_mask])
    ate_values[category] = weighted_avg_outcome
    total += ate_values[category]
ate = total / 3
print(f'Estimated ATE: {ate}')
print(f'Estimated ATE_values: {ate_values}')

confusion3d(propensity_model, X_test, y_test, category_mapping)
plot_accuracy(history)
plot_loss(history)