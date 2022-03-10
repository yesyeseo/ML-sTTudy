# 4. 신경망 학습

신경망을 학습할 때 정확도를 지표로 삼아서는 안 됨 → 매개변수의 미분이 대부분의 장소에서 0이 됨

연속 함수: 출력(값)이 연속적으로 변함 & 곡선의 기울기도 연속적으로 변함 → 시그모이드의 경우, 어느 장소라도 미분값이 0이 되지 않음 → 기울기가 0이 되지 않으므로 신경망이 올바르게 학습될 수 있음

## 4.2. 손실 함수

### 4.2.3 미니 배치 학습

- 훈련 데이터에 대한 손실 함수의 값 구함 → 그 값을 최소한으로 하는 매개 변수 탐색
    - **모든** 훈련 데이터를 대상으로 하는 손실 함수의 값 구해야 함 → 시간이 오래 걸림 → 훈련 데이터로부터 **일부**만 골라 **전체의 근사치**로 이용
    - 고른 그 일부를 `mini-batch`라고 함.
        - ex) 60,000장의 훈련 데이터 중 100장 무작위로 뽑아 100장만을 사용하여 학습: `미니배치 학습`
    - 전체 훈련 데이터의 대표로 무작위로 선택한 작은 미니 배치를 이용

```python
# mnist dataset
# x_train.shape = (60000, 784)
# y_train.shape = (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
# 훈련 데이터에서 무작위로 10장을 빼냄
# 지정한 범위의 수 중 무작위로 원하는 개수만 꺼낼 수 있음
# 함수가 출력할 배열을 뽑아낼 데이터의 인덱스로 사용하면 됨
x_batch = x_train[batch_mask]
t_train = t_train[batch_mask]
```

---

### 4.2.4 (배치용) 교차 엔트로피 오차 구현

**정답 레이블이 원-핫 인코딩일 때**

```python
def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size) # 정답 레이블 
		y = y.reshape(1, y.size) # 신경망의 출력

	batch_size = y.shape[0]
	return -np.sum(t * np.log(y + 1e-7)) / batch_size 
	# 배치의 크기로 나누어 정규화함, 이미지 1장당 평균의 교차 엔트로피 오차 계산
```

 

**정답 레이블이 숫자 레이블로 주어졌을 때**

- t가 0인 원소는 교차 엔트로피 오차도 0임 → 그 계산은 무시해도 됨
- 정답에 해당하는 신경망의 출력**만**으로 교차 엔트로피 오차 계산 가능

```python
def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size) # 정답의 숫자 레이블
		y = y.reshape(1, y.size) # 신경망의 출력

	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arrange(batch_size), t] + 1e-7)) / batch_size
	# 정답에 해당하는 신경망의 출력만으로 교차 엔트로피 오차 계산
```

- `np.arrange(batch_size)` : [0, batch_size-1] 배열 생성
- `y[np.arrange(batch_size), t]` : 각 데이터의 정답 레이블에 해당하는 신경망의 출력 추출
    - ex. [y[0, 2], y[1, 7] ... y[4, 5]] 처럼

---

## 4.3. 수치 미분

- 미분: 한순간의 변화량
- 주의해야 할 점
    - 반올림 오차 (rounding error) : 작은 값이 생략됨 → 최종 계산 결과에 오차 발생
        - h를 10^(-4) 정도의 값으로 사용하면 좋은 결과를 얻을 수도 있음
    - 함수 f의 차분 : h를 무한히 0으로 좁히는 것이 불가능하므로 생기는 한계점 → **중심 차분**으로 개선

```python
def numercial_diff(f, x):
	h = 1e-4 # 0.0001
	return (f(x+h) - f(x-h)) / (2*h) # 중심 차분
	# return (f(x + h) - f(x)) / h - 실재 접선 기울기와 일치하지 않는다는 한계 있음
```

---

## 4.4. 기울기(gradient)

**모든 변수의 편미분을 벡터로 정리한 것**

```python
def numercial_gradient(f, x):
	h = 1e-4 # 0.0001
	grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성함

	for idx in range(x.sizE):
		tmp_val = x[idx]
		# f(x+h) 계산
		x[idx] = tmp_val + h
		
		# f(x-h) 계산
		x[idx] = tmp_val - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val # 값 복원
	
	return grad
```