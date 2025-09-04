import numpy as np

class Node:
    def __init__(self, value, parents=(), op=''):
        self.value = np.array(value, dtype=float)
        self.parents = parents
        self.op = op
        self.grad = 0.0

    def backward(self, grad=1.0):
        self.grad += grad
        if self.op == 'add':
            self.parents[0].backward(grad)
            self.parents[1].backward(grad)
        elif self.op == 'sub':
            self.parents[0].backward(grad)
            self.parents[1].backward(-grad)
        elif self.op == 'mul':
            self.parents[0].backward(grad * self.parents[1].value)
            self.parents[1].backward(grad * self.parents[0].value)
        elif self.op == 'div':
            self.parents[0].backward(grad / self.parents[1].value)
            self.parents[1].backward(-grad * self.parents[0].value / (self.parents[1].value**2))
        elif self.op == 'pow':
            base, exp = self.parents
            self.parents[0].backward(grad * exp.value * (base.value**(exp.value-1)))
        elif self.op == 'exp':
            self.parents[0].backward(grad * np.exp(self.parents[0].value))
        elif self.op == 'log':
            self.parents[0].backward(grad / self.parents[0].value)
        elif self.op == 'sin':
            self.parents[0].backward(grad * np.cos(self.parents[0].value))
        elif self.op == 'cos':
            self.parents[0].backward(-grad * np.sin(self.parents[0].value))
        elif self.op == 'tan':
            self.parents[0].backward(grad / (np.cos(self.parents[0].value)**2))
        elif self.op == 'dot':
            A, B = self.parents
            self.parents[0].backward(grad @ B.value.T)
            self.parents[1].backward(A.value.T @ grad)
        elif self.op == 'sigmoid':
            s = 1/(1+np.exp(-self.parents[0].value))
            self.parents[0].backward(grad * s * (1-s))
        elif self.op == 'tanh':
            t = np.tanh(self.parents[0].value)
            self.parents[0].backward(grad * (1-t**2))
        elif self.op == 'relu':
            self.parents[0].backward(grad * (self.parents[0].value > 0))

# ---------- Helper functions ----------
def add(a,b): return Node(a.value+b.value,(a,b),'add')
def sub(a,b): return Node(a.value-b.value,(a,b),'sub')
def mul(a,b): return Node(a.value*b.value,(a,b),'mul')
def div(a,b): return Node(a.value/b.value,(a,b),'div')
def pow(a,b): return Node(a.value**b.value,(a,b),'pow')
def exp(a): return Node(np.exp(a.value),(a,),'exp')
def log(a): return Node(np.log(a.value),(a,),'log')
def sin(a): return Node(np.sin(a.value),(a,),'sin')
def cos(a): return Node(np.cos(a.value),(a,),'cos')
def tan(a): return Node(np.tan(a.value),(a,),'tan')
def dot(a,b): return Node(a.value@b.value,(a,b),'dot')
def sigmoid(a): return Node(1/(1+np.exp(-a.value)),(a,),'sigmoid')
def tanh(a): return Node(np.tanh(a.value),(a,),'tanh')
def relu(a): return Node(np.maximum(0,a.value),(a,),'relu')

# ---------- Example ----------
x = Node(2.0)
y = Node(3.0)

z = add(mul(x,y),sin(x))   # z = x*y + sin(x)
z.backward()

print("z:", z.value)
print("dz/dx:", x.grad)
print("dz/dy:", y.grad)

A = Node(np.array([[1,2],[3,4]]))
B = Node(np.array([[2],[1]]))
out = dot(A,B)   # matrix dot product
out.backward(np.ones_like(out.value))
print("dot result:\n", out.value)
print("dOut/dA:\n", A.grad)
print("dOut/dB:\n", B.grad)

#GBM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Load dataset (example: Titanic dataset from local CSV)
df = pd.read_csv("titanic.csv")

# Select features & target
X = df[["Pclass", "Age", "Fare"]].fillna(0)   # example numeric features
y = df["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

print("True labels:     ", list(y_test[:10]))
print("Predicted labels:", list(y_pred[:10]))
print("Accuracy:", clf.score(X_test, y_test))
