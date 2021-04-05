# knn-from-scratch

In this program, the KNN algorithm is written from scratch using the inverse distance weighted method and predicts the classes of test data. Then the results of KNN written from scratch and scikit learn are compared and the decision boundaries are plotted.


### Classification Results

Comparing both Sklearn and from Scratch KNN implementations, all of the accuracy score and error count values are the equal except k = 1. The reason why the results are different when k = 1 is that there are two points in different classes at the same distance. Since the program chooses the highest voted class in the KNN algorithm that we coded it chooses the wrong class in this test data set.

<p align="center">
<img width="276" alt="Ekran Resmi 2021-03-15 20 41 06" src="https://user-images.githubusercontent.com/52889449/111196831-cf250880-85ce-11eb-9b6f-e97302f9eb73.png">
</p>

### Scatter Plot

Here is the scatter plot visualization of the training data set.

<p align="center">
<img width="433" alt="Ekran Resmi 2021-03-15 20 42 05" src="https://user-images.githubusercontent.com/52889449/111196939-ea901380-85ce-11eb-9355-4853e44de3aa.png">
</p>

### Decision Boundaries

These figures show the decision boundaries plot in k = 1 and k = 25 for each Sklearn and Scratch KNN implementations. As can be observed, there is no difference between the decision boundaries of the Sklearn KNN and the KNN that we coded from scratch.

<p align="center">
<img width="431" alt="Ekran Resmi 2021-03-15 20 42 45" src="https://user-images.githubusercontent.com/52889449/111197162-2e831880-85cf-11eb-8a42-ccb202f09d18.png">
<img width="431" alt="Ekran Resmi 2021-03-15 20 43 05" src="https://user-images.githubusercontent.com/52889449/111197168-2fb44580-85cf-11eb-9a87-da2ddc29e23e.png">
<img width="431" alt="Ekran Resmi 2021-03-15 20 43 25" src="https://user-images.githubusercontent.com/52889449/111197171-317e0900-85cf-11eb-8643-4b07eb3ef345.png">
<img width="433" alt="Ekran Resmi 2021-03-15 20 43 44" src="https://user-images.githubusercontent.com/52889449/111197180-3478f980-85cf-11eb-9b14-96ebf9273543.png">
</p>

