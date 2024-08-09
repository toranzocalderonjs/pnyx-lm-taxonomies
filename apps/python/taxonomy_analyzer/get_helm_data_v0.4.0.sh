#/bin/sh
LEADERBOARD_VERSION=v0.4.0
curl -O https://nlp.stanford.edu/helm/archives/$LEADERBOARD_VERSION-instances.zip
mkdir -p ./helm/benchmark_output/runs/$LEADERBOARD_VERSION
unzip $LEADERBOARD_VERSION-instances.zip -d ./helm/benchmark_output/runs/$LEADERBOARD_VERSION
rm $LEADERBOARD_VERSION-instances.zip