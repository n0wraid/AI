from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification

# ��������� ��������� ������ ��� �������
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# ���������� ������ �� ��������� � �������� ������
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ����������� ������� ���������������
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(probability=True, random_state=42)

# �������� ������� ���������������
rf_clf.fit(X_train, y_train)
gb_clf.fit(X_train, y_train)
svm_clf.fit(X_train, y_train)

# �������� ������������� ����������
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('gb', gb_clf), ('svm', svm_clf)],
    voting='soft'  # 'soft' ��� ������������� ������������ �������
)

# �������� ����������
voting_clf.fit(X_train, y_train)

# ������������ �� �������� ������ � ������
for clf, label in zip([rf_clf, gb_clf, svm_clf, voting_clf], ['Random Forest', 'Gradient Boosting', 'SVM', 'Voting']):
    y_pred_individual = clf.predict(X_test)
    
    # ������ ��������
    accuracy = accuracy_score(y_test, y_pred_individual)
    print(f'{label} Accuracy: {accuracy}')

    # ������ MSE
    mse = mean_squared_error(y_test, y_pred_individual)
    print(f'{label} Mean Squared Error: {mse}\n')
