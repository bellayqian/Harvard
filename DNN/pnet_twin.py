from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import AUC
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from twin_preprocessing import twin_preprocessing, merge_into_one, plot_loss, plot_accuracy, confusion



# Creating the PropensityNet model
def create_propensity_net(input_shape, output_units=1, hidden_layers=[94, 1175, 293,94], dropout_rate=0.3, l2_reg=0.0001):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    # Adding hidden layers
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.BatchNormalization())
    
    # Output layer
    model.add(layers.Dense(output_units, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])#, AUC(name='auc')])
    return model

merged_data = pd.read_csv('Harvard/MIT67900/proj/merged_outcome_twin_unscaled.csv')
# hidden_layers=[47, 1175, 293, 587, 1175, 376, 94, 188]
# Split the features and labels from the dataset
# X_features = merged_data.drop(['T', 'Outcome'], axis=1)  # Remove the label columns to isolate the features
scaler = StandardScaler()
X = scaler.fit_transform(merged_data.drop(['tobacco', 'Outcome', 'bord_0','bord_1'], axis=1))

y = merged_data['tobacco']  # The treatment column is the label for the PropensityNet
# categorical_cols = X.select_dtypes(include=['object', 'category']).columns
# continuous_cols = X.select_dtypes(include=['int64', 'float64']).columns

# # Preprocessing for continuous data
# continuous_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),  # or median, if outliers are present
#     ('scaler', MinMaxScaler())  # or StandardScaler()
# ])

# # Preprocessing for categorical data
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Bundle preprocessing for numerical and categorical data
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', continuous_transformer, continuous_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_train = preprocessor.fit_transform(X_train)
#X_test = preprocessor.transform(X_test)
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

propensity_scores = propensity_model.predict(X).ravel()
print(propensity_scores)
# Add the propensity scores to your dataset
merged_data['Propensity_Score'] = propensity_scores

# Step 2: Match or weight the samples using propensity scores
# Here you would use a method to create matched pairs or weights based on the scores.
# This step is quite involved and may require additional libraries such as `causalml` or `pyMatch`.
# For simplicity, let's assume you're creating a basic weight for IPTW.
weights = (merged_data['tobacco'] / propensity_scores) + ((1 - merged_data['tobacco']) / (1 - propensity_scores))

# Step 3: Estimate ATE
# Using the weights, calculate the weighted average outcome for both treatment and control groups
treated_outcome = (merged_data['Outcome'] * weights * merged_data['tobacco']).sum() / (weights * merged_data['tobacco']).sum()
control_outcome = (merged_data['Outcome'] * weights * (1 - merged_data['tobacco'])).sum() / (weights * (1 - merged_data['tobacco'])).sum()

# The ATE is the difference between these two averages
ate = treated_outcome - control_outcome
print(f'Estimated ATE: {ate}')

plot_accuracy(history)
plot_loss(history)

# Confusion Matrix
confusion(propensity_model, X_test, y_test)


