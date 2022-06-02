load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# To update TensorFlow to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
# http_archive(
#    name = "org_tensorflow",
#    sha256 = "29bf0018cef471649424cc76c5ea8601464252ce7a0e88c69a46aa9a0030c783",
#    strip_prefix = "tensorflow-128e8923b97ee054b8f5d4890cd8f41b06b74d7d",
#    urls = [
#        "https://github.com/tensorflow/tensorflow/archive/128e8923b97ee054b8f5d4890cd8f41b06b74d7d.tar.gz",
#    ],
#)

git_repository(
    name = "org_tensorflow",
    branch = "jaxlibv0.3.15/sdk-release-3.0-lean-jax",
    remote = "git@github.com:graphcore/tensorflow-jax-poc.git"
)

# For development, one often wants to make changes to the TF repository as well
# as the JAX repository. You can override the pinned repository above with a
# local checkout by either:
# a) overriding the TF repository on the build.py command line by passing a flag
#    like:
#    python build/build.py --bazel_options=--override_repository=org_tensorflow=/path/to/tensorflow
#    or
# b) by commenting out the http_archive above and uncommenting the following:
# local_repository(
#    name = "org_tensorflow",
#    path = "/path/to/tensorflow",
# )

load("//third_party/pocketfft:workspace.bzl", pocketfft = "repo")
pocketfft()

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()
