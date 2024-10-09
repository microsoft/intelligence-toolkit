export COGNITO_APP_CLIENT_SECRET=37q79h2fks4escfpcgvjiomsorr94ofq0vgt9dk2s62m0hg23hu
export COGNITO_POOL_ID=us-east-1_X9ZSyUfDr
export COGNITO_APP_CLIENT_ID=5grfpne9si415rrb5ruh5bqpd5
export POLARS_SKIP_CPU_CHECK=true

GIT_DELTA_URL=https://github.com/dandavison/delta/releases/download/0.18.2/delta-0.18.2-x86_64-unknown-linux-gnu.tar.gz
# GIT_DELTA_URL=https://github.com/dandavison/delta/releases/download/0.18.2/git-delta_0.18.2_amd64.deb

curl -L $GIT_DELTA_URL -o git-delta.tar.gz
tar -xvf git-delta.tar.gz
mv delta-*/delta ~/.local/bin
rm -rf git-delta*tar.gz delta-*
