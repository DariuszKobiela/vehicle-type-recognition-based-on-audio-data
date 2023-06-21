# vehicle-type-recognition-based-on-audio-data

The goal of the project was to create a neural
network model, which basing on the input data in the form of
an audio files would be able to recognize the type of the vehicle
passing. The project was a continuation of the research carried
out by A. Czy ̇zewski, A. Kurowski and S. Zaporowski [[1]](#1).
In the shown approach there is combined well extracted and
denoised feature vector in the form of mel spectrograms with
advanced CNN (Convolutional Neural Network) model in order
to outperform existing networks for vehicle sound classification.
The obtained results confirms, that the best model, imple-
mented in pytorch with the usage of mel-spectrograms and with
the augmented data, achieves the metrics values on the level
of accuracy = 0.875 and f1-score = 0.88. It was decided to use
time shift and masking out with horizontal and vertical black
bars (time and frequency) as augmentation technique. Therefore,
comparing to the solution shown in [[1]](#1), where SVC (Support
Vector Classifier) model achieved the weighted accuracy at the
level of 0.93 and weighted f1-score = 0.54, the proposed CNN
model significantly beats the SVM in the terms of f1-score.
The accuracy of CNN model is slightly lower, however, the
approach showed in [[1]](#1) takes into consideration only 2 classes
(car and truck), while this approach takes 3 classes (car, truck
and motorcycle). It was also showed that data preprocessing for
audio data using mel-spectrograms gives better results than using
MFCCs(Mel-frequency cepstral coefficients).

`[embed]https://github.com/DariuszKobiela/vehicle-type-recognition-based-on-audio-data/2023_AI_TECH_research_project_poster.pdf[/embed]`

## References
<a id="1">[1]</a> 
Czyżewski, A., Kurowski, A., & Zaporowski, S. (2019). 
Application of autoencoder to traffic noise analysis. 
Journal of the Acoustical Society of America, 146, 2958-2958. 
https://doi.org/10.1121/1.5137275