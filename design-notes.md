

Q: How to install?
A: Ideally it would be via a pip package, thats simplest. In this case though there just seem to be the apt. Right now I have choosen to do source install, and then I have added additional functions. This feels a bit silly? The alternative is to wrap tf_transformations in order to make it easy to upgrade indepdently... what I have done is not terrible either though, I am not expecting this package to update very often...

The basic idea is that the msg data types should implement some of these spatial transforms, so I need helpers below to provide