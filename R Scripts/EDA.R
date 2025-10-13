library(corrplot)
library(ggplot2)

df1 = read.csv("/Users/siddharthhaveliwala/Documents/Fall 2023/IE 500 - PM & DA/Project/Dataset/playlist_2010to2022.csv")

na.count<- sum(is.na(df1))
na.count

df1[is.na(df1)]
df1 <- na.omit(df1)
sum(is.na(df1))

install.packages("purrr")
library(purrr)
map_int(df1, function(x) sum(x=="[]"))

library(tidyverse)
df12 <- select(df1, -c(playlist_url, track_id, artist_id, artist_genres))
View(df12)
head(df12)
str(df12)

corrln <- cor(df12[, c(7:19)])
corrln
#Draw the correlation plot
corrplot(corrln, outline = TRUE, addgrid.col = "lightblue", type="upper", method = "number")
mtext("Music Features Correlation Plot", at=5, line=-0.5, cex=1.5)

#df12$track_name[df12$valence==0.0377]
#min(df12$valence)
# Set up a multi-panel layout with 3 rows and 4 columns
par(mfrow=c(3, 4))
# Plot histograms for each column
hist(df12$danceability, main="Danceability", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$energy, main="Energy", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$key, main="Key", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$loudness, main="Loudness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$mode, main="Mode", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$speechiness, main="Speechiness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$acousticness, main="Acousticness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$instrumentalness, main="Instrumentalness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$liveness, main="Liveness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$valence, main="Valence", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$tempo, main="Tempo", col="skyblue", border="black", xlab="Values", ylab="Frequency")
hist(df12$time_signature, main="Time Signature", col="skyblue", border="black", xlab="Values", ylab="Frequency")
str(df12)
# Set up a multi-panel layout with 3 rows and 4 columns
#par(mfrow=c(3, 4))
# Create boxplots for each column
boxplot(df12$danceability, main="Danceability", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$energy, main="Energy", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$key, main="Key", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$loudness, main="Loudness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$mode, main="Mode", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$speechiness, main="Speechiness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$acousticness, main="Acousticness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$instrumentalness, main="Instrumentalness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$liveness, main="Liveness", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$valence, main="Valence", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$tempo, main="Tempo", col="skyblue", border="black", xlab="Values", ylab="Frequency")
boxplot(df12$time_signature, main="Time Signature", col="skyblue", border="black", xlab="Values", ylab="Frequency")

dev.off()
par(mfrow=c(2,2))
plot(df12$valence, df12$energy,
     xlab = "Valence",
     ylab = "Energy",
     main = "Scatter Plot of Valence vs. Energy",
     pch = 19, # Point shape
     col = "blue" # Point color
)
abline(lm(df12$energy ~ df12$valence), col = "red", lwd = 2)
plot(df12$loudness, df12$energy,
     xlab = "Loudness",
     ylab = "Energy",
     main = "Scatter Plot of Loudness vs. Energy",
     pch = 19, # Point shape
     col = "red" # Point color
)
abline(lm(df12$energy ~ df12$loudness), col = "blue", lwd = 2)
plot(df12$valence, df12$danceability,
     xlab = "Valence",
     ylab = "Danceability",
     main = "Scatter Plot of Valence vs. Danceability",
     pch = 19, # Point shape
     col = "pink" # Point color
)
abline(lm(df12$valence ~ df12$danceability), col = "blue", lwd = 2)
plot(df12$acousticness, df12$energy,
     xlab = "Acousticness",
     ylab = "Energy",
     main = "Scatter Plot of Acousticness vs. Energy",
     pch = 19, # Point shape
     col = "green" # Point color
)
abline(lm(df12$acousticness ~ df12$energy), col = "red", lwd = 2)

library(datasets)

library(PerformanceAnalytics)

chart.Correlation(df12[, 7:19], histogram = TRUE, method = "pearson")

pairs(~ danceability + energy + key + loudness + mode + speechiness + acousticness + instrumentalness + liveness +
        valence + tempo + time_signature,
      data = df12,
      col = c("black","orange","purple"),
      upper.panel = NULL,    # Correlation panel
      lower.panel = panel.smooth,
      #diag.panel = panel.hist,
      pch = 18)


summary(df12)

df13 <- select(df12, -c(year, track_name, track_popularity, album, artist_name, artist_popularity))

scaled_data <- scale(df13)

pca_result <- prcomp(scaled_data)

variance_explained <- pca_result$sdev^2
variance_explained_ratio <- variance_explained / sum(variance_explained)

variance_explained

n <- 10
selected_features <- df13[, pca_result$rotation[, 1:n]]

selected_features

reduced_data <- pca_result$x
reduced_data

