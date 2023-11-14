from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification

# Генерация случайных данных для примера
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение базовых классификаторов
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(probability=True, random_state=42)

# Обучение базовых классификаторов
rf_clf.fit(X_train, y_train)
gb_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

# Создание квазилинейной композиции
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('gb', gb_clf), ('svm', svm_clf)],
    voting='soft'  # 'soft' для использования вероятностей классов
)

# Обучение композиции
voting_clf.fit(X_train, y_train)

# Предсказание на тестовых данных и оценка
for clf, label in zip([rf_clf, gb_clf, svm_clf, voting_clf], ['Random Forest', 'Gradient Boosting', 'SVM', 'Voting']):
    y_pred_individual = clf.predict(X_test)
    
    # Оценка точности
    accuracy = accuracy_score(y_test, y_pred_individual)
    print(f'{label} Accuracy: {accuracy}')

    # Оценка MSE
    mse = mean_squared_error(y_test, y_pred_individual)
    print(f'{label} Mean Squared Error: {mse}\n')
