# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Classes and functions used to construct graphs."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import linecache
import sys


# Temporary global switch determining if we should enable the work-in-progress
# calls to the C API. Currently disabled by default but can be manually enabled
# e.g. in tests. This will be removed once all functionality is supported and
# there's no performance penalty with it enabled.
#
# TODO(skyewm) before we can remove this:
# - functions
# - import_graph_def() incrementally adds inputs to ops (i.e. creates an
#   Operation and then calls _add_input()). The current code requires that all
#   inputs be specified when creating the Operation (since we call
#   TF_FinishOperation()).
# - ops_test.py (and others?) create unregistered op types
# - while loop
# - performance (e.g. delete/refactor redundant Python functionality, switch to
#   new session API)
_USE_C_API = False


def _override_helper(clazz_object, operator, func):
  """Overrides (string) operator on Tensors to call func.

  Args:
    clazz_object: the class to override for; either Tensor or SparseTensor.
    operator: the string name of the operator to override.
    func: the function that replaces the overridden operator.

  Raises:
    ValueError: If operator has already been overwritten,
      or if operator is not allowed to be overwritten.
  """
  existing = getattr(clazz_object, operator, None)
  if existing is not None:
    # Check to see if this is a default method-wrapper or slot wrapper which
    # will be true for the comparison operators.
    if not isinstance(existing, type(object.__lt__)):
      raise ValueError("operator %s cannot be overwritten again on class %s." %
                       (operator, clazz_object))
  if operator not in Tensor.OVERLOADABLE_OPERATORS:
    raise ValueError("Overriding %s is disallowed" % operator)
  setattr(clazz_object, operator, func)


def _convert_stack(stack, include_func_start_lineno=False):
  """Converts a stack extracted using _extract_stack() to a traceback stack.

  Args:
    stack: A list of n 5-tuples,
      (filename, lineno, name, frame_globals, func_start_lineno).
    include_func_start_lineno: True if function start line number should be
      included as the 5th entry in return tuples.

  Returns:
    A list of n 4-tuples or 5-tuples
    (filename, lineno, name, code, [optional: func_start_lineno]), where the
    code tuple element is calculated from the corresponding elements of the
    input tuple.
  """
  ret = []
  for filename, lineno, name, frame_globals, func_start_lineno in stack:
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, frame_globals)
    if line:
      line = line.strip()
    else:
      line = None
    if include_func_start_lineno:
      ret.append((filename, lineno, name, line, func_start_lineno))
    else:
      ret.append((filename, lineno, name, line))
  return ret


# pylint: disable=line-too-long
def _extract_stack():
  """A lightweight re-implementation of traceback.extract_stack.

  NOTE(mrry): traceback.extract_stack eagerly retrieves the line of code for
    each stack frame using linecache, which results in an abundance of stat()
    calls. This implementation does not retrieve the code, and any consumer
    should apply _convert_stack to the result to obtain a traceback that can
    be formatted etc. using traceback methods.

  Returns:
    A list of 5-tuples
    (filename, lineno, name, frame_globals, func_start_lineno) corresponding to
    the call stack of the current thread.
  """
  # pylint: enable=line-too-long
  try:
    raise ZeroDivisionError
  except ZeroDivisionError:
    f = sys.exc_info()[2].tb_frame.f_back
  ret = []
  while f is not None:
    lineno = f.f_lineno
    co = f.f_code
    filename = co.co_filename
    name = co.co_name
    frame_globals = f.f_globals
    func_start_lineno = co.co_firstlineno
    ret.append((filename, lineno, name, frame_globals, func_start_lineno))
    f = f.f_back
  ret.reverse()
  return ret


def _as_graph_element(obj):
  """Convert `obj` to a graph element if possible, otherwise return `None`.

  Args:
    obj: Object to convert.

  Returns:
    The result of `obj._as_graph_element()` if that method is available;
        otherwise `None`.
  """
  conv_fn = getattr(obj, "_as_graph_element", None)
  if conv_fn and callable(conv_fn):
    return conv_fn()
  return None


_TENSOR_LIKE_TYPES = tuple()


def is_dense_tensor_like(t):
  """EXPERIMENTAL: Returns true if `t` implements the tensor interface.

  See `register_dense_tensor_like_type()` for the current definition of a
  "tensor-like type".

  Args:
    t: An object.

  Returns:
    True iff `t` is an instance of one of the registered "tensor-like" types.
  """
  return isinstance(t, _TENSOR_LIKE_TYPES)


def register_dense_tensor_like_type(tensor_type):
  """EXPERIMENTAL: Registers `tensor_type` as implementing the tensor interface.

  A "tensor-like type" can represent a single dense tensor, and implements
  the `name` and `dtype` properties.

  Args:
    tensor_type: A type implementing the tensor interface.

  Raises:
    TypeError: If `tensor_type` does not implement the tensor interface.
  """
  try:
    if not isinstance(tensor_type.name, property):
      raise TypeError("Type %s does not define a `name` property")
  except AttributeError:
    raise TypeError("Type %s does not define a `name` property")
  try:
    if not isinstance(tensor_type.dtype, property):
      raise TypeError("Type %s does not define a `dtype` property")
  except AttributeError:
    raise TypeError("Type %s does not define a `dtype` property")
  # We expect this list to be small, so choose quadratic complexity
  # for registration, so that we have a tuple that can be used for
  # more efficient `isinstance` checks later.
  global _TENSOR_LIKE_TYPES
  _TENSOR_LIKE_TYPES = tuple(list(_TENSOR_LIKE_TYPES) + [tensor_type])


# NOTE(ebrevdo): Do not subclass this.  If you do, I will break you on purpose.
class _TensorLike(object):
  """Internal cls for grouping Tensor, SparseTensor, ..., for is_instance."""
  pass


class Tensor(_TensorLike):
  """Represents one of the outputs of an `Operation`.

  A `Tensor` is a symbolic handle to one of the outputs of an
  `Operation`. It does not hold the values of that operation's output,
  but instead provides a means of computing those values in a
  TensorFlow @{tf.Session}.

  This class has two primary purposes:

  1. A `Tensor` can be passed as an input to another `Operation`.
     This builds a dataflow connection between operations, which
     enables TensorFlow to execute an entire `Graph` that represents a
     large, multi-step computation.

  2. After the graph has been launched in a session, the value of the
     `Tensor` can be computed by passing it to
     @{tf.Session.run}.
     `t.eval()` is a shortcut for calling
     `tf.get_default_session().run(t)`.

  In the following example, `c`, `d`, and `e` are symbolic `Tensor`
  objects, whereas `result` is a numpy array that stores a concrete
  value:

  ```python
  # Build a dataflow graph.
  c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
  e = tf.matmul(c, d)

  # Construct a `Session` to execute the graph.
  sess = tf.Session()

  # Execute the graph and store the value that `e` represents in `result`.
  result = sess.run(e)
  ```
  """

  # List of Python operators that we allow to override.
  OVERLOADABLE_OPERATORS = {
      # Binary.
      "__add__",
      "__radd__",
      "__sub__",
      "__rsub__",
      "__mul__",
      "__rmul__",
      "__div__",
      "__rdiv__",
      "__truediv__",
      "__rtruediv__",
      "__floordiv__",
      "__rfloordiv__",
      "__mod__",
      "__rmod__",
      "__lt__",
      "__le__",
      "__gt__",
      "__ge__",
      "__and__",
      "__rand__",
      "__or__",
      "__ror__",
      "__xor__",
      "__rxor__",
      "__getitem__",
      "__pow__",
      "__rpow__",
      # Unary.
      "__invert__",
      "__neg__",
      "__abs__",
      "__matmul__",
      "__rmatmul__"
  }

  def __init__(self, op, value_index, dtype):
    """Creates a new `Tensor`.

    Args:
      op: An `Operation`. `Operation` that computes this tensor.
      value_index: An `int`. Index of the operation's endpoint that produces
        this tensor.
      dtype: A `DType`. Type of elements stored in this tensor.

    Raises:
      TypeError: If the op is not an `Operation`.
    """
    if not isinstance(op, Operation):
      raise TypeError("op needs to be an Operation: %s" % op)
    self._op = op
    self._value_index = value_index
    self._dtype = dtypes.as_dtype(dtype)
    self._shape = tensor_shape.unknown_shape()
    # List of operations that use this Tensor as input.  We maintain this list
    # to easily navigate a computation graph.
    self._consumers = []

    # Attributes used for C++ shape inference. Not inspected, only forwarded.
    self._handle_shape = tensor_shape_pb2.TensorShapeProto()
    self._handle_dtype = types_pb2.DT_INVALID

  @property
  def op(self):
    """The `Operation` that produces this tensor as an output."""
    return self._op

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self._dtype

  @property
  def graph(self):
    """The `Graph` that contains this tensor."""
    return self._op.graph

  @property
  def name(self):
    """The string name of this tensor."""
    if not self._op.name:
      raise ValueError("Operation was not named: %s" % self._op)
    return "%s:%d" % (self._op.name, self._value_index)

  @property
  def device(self):
    """The name of the device on which this tensor will be produced, or None."""
    return self._op.device

  @property
  def shape(self):
    """Returns the `TensorShape` that represents the shape of this tensor.

    The shape is computed using shape inference functions that are
    registered in the Op for each `Operation`.  See
    @{tf.TensorShape}
    for more details of what a shape represents.

    The inferred shape of a tensor is used to provide shape
    information without having to launch the graph in a session. This
    can be used for debugging, and providing early error messages. For
    example:

    ```python
    c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    print(c.shape)
    ==> TensorShape([Dimension(2), Dimension(3)])

    d = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    print(d.shape)
    ==> TensorShape([Dimension(4), Dimension(2)])

    # Raises a ValueError, because `c` and `d` do not have compatible
    # inner dimensions.
    e = tf.matmul(c, d)

    f = tf.matmul(c, d, transpose_a=True, transpose_b=True)

    print(f.shape)
    ==> TensorShape([Dimension(3), Dimension(4)])
    ```

    In some cases, the inferred shape may have unknown dimensions. If
    the caller has additional information about the values of these
    dimensions, `Tensor.set_shape()` can be used to augment the
    inferred shape.

    Returns:
      A `TensorShape` representing the shape of this tensor.

    """
    return self._shape

  def _shape_as_list(self):
    if self._shape.ndims is not None:
      return [dim.value for dim in self._shape.dims]
    else:
      return None

  def get_shape(self):
    """Alias of Tensor.shape."""
    return self.shape

  def set_shape(self, shape):
    """Updates the shape of this tensor.

    This method can be called multiple times, and will merge the given
    `shape` with the current shape of this tensor. It can be used to
    provide additional information about the shape of this tensor that
    cannot be inferred from the graph alone. For example, this can be used
    to provide additional information about the shapes of images:

    ```python
    _, image_data = tf.TFRecordReader(...).read(...)
    image = tf.image.decode_png(image_data, channels=3)

    # The height and width dimensions of `image` are data dependent, and
    # cannot be computed without executing the op.
    print(image.shape)
    ==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])

    # We know that each image in this dataset is 28 x 28 pixels.
    image.set_shape([28, 28, 3])
    print(image.shape)
    ==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])
    ```

    Args:
      shape: A `TensorShape` representing the shape of this tensor.

    Raises:
      ValueError: If `shape` is not compatible with the current shape of
        this tensor.
    """
    self._shape = self._shape.merge_with(shape)

  @property
  def value_index(self):
    """The index of this tensor in the outputs of its `Operation`."""
    return self._value_index

  def consumers(self):
    """Returns a list of `Operation`s that consume this tensor.

    Returns:
      A list of `Operation`s.
    """
    return self._consumers

  def _add_consumer(self, consumer):
    """Add a consumer to this tensor.

    Args:
      consumer: an Operation.

    Raises:
      TypeError: if the consumer is not an Operation.
    """
    if not isinstance(consumer, Operation):
      raise TypeError("Consumer must be an Operation: %s" % consumer)
    self._consumers.append(consumer)

  def _as_node_def_input(self):
    """Return a value to use for the NodeDef "input" attribute.

    The returned string can be used in a NodeDef "input" attribute
    to indicate that the NodeDef uses this Tensor as input.

    Raises:
      ValueError: if this Tensor's Operation does not have a name.

    Returns:
      a string.
    """
    if not self._op.name:
      raise ValueError("Operation was not named: %s" % self._op)
    if self._value_index == 0:
      return self._op.name
    else:
      return "%s:%d" % (self._op.name, self._value_index)

  def _as_tf_output(self):
    assert self.op._c_op  # pylint: disable=protected-access
    tf_output = c_api.TF_Output()
    tf_output.oper = self.op._c_op  # pylint: disable=protected-access
    tf_output.index = self.value_index
    return tf_output

  def __str__(self):
    return "Tensor(\"%s\"%s%s%s)" % (
        self.name,
        (", shape=%s" % self.get_shape())
        if self.get_shape().ndims is not None else "",
        (", dtype=%s" % self._dtype.name) if self._dtype else "",
        (", device=%s" % self.device) if self.device else "")

  def __repr__(self):
    return "<tf.Tensor '%s' shape=%s dtype=%s>" % (
        self.name, self.get_shape(), self._dtype.name)

  def __hash__(self):
    # Necessary to support Python's collection membership operators
    return id(self)

  def __eq__(self, other):
    # Necessary to support Python's collection membership operators
    return id(self) == id(other)

  # NOTE(mrry): This enables the Tensor's overloaded "right" binary
  # operators to run when the left operand is an ndarray, because it
  # accords the Tensor class higher priority than an ndarray, or a
  # numpy matrix.
  # TODO(mrry): Convert this to using numpy's __numpy_ufunc__
  # mechanism, which allows more control over how Tensors interact
  # with ndarrays.
  __array_priority__ = 100

  @staticmethod
  def _override_operator(operator, func):
    _override_helper(Tensor, operator, func)

  def __iter__(self):
    """Dummy method to prevent iteration. Do not call.

    NOTE(mrry): If we register __getitem__ as an overloaded operator,
    Python will valiantly attempt to iterate over the Tensor from 0 to
    infinity.  Declaring this method prevents this unintended
    behavior.

    Raises:
      TypeError: when invoked.
    """
    raise TypeError("'Tensor' object is not iterable.")

  def __bool__(self):
    """Dummy method to prevent a tensor from being used as a Python `bool`.

    This overload raises a `TypeError` when the user inadvertently
    treats a `Tensor` as a boolean (e.g. in an `if` statement). For
    example:

    ```python
    if tf.constant(True):  # Will raise.
      # ...

    if tf.constant(5) < tf.constant(7):  # Will raise.
      # ...
    ```

    This disallows ambiguities between testing the Python value vs testing the
    dynamic condition of the `Tensor`.

    Raises:
      `TypeError`.
    """
    raise TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed. "
                    "Use `if t is not None:` instead of `if t:` to test if a "
                    "tensor is defined, and use TensorFlow ops such as "
                    "tf.cond to execute subgraphs conditioned on the value of "
                    "a tensor.")

  def __nonzero__(self):
    """Dummy method to prevent a tensor from being used as a Python `bool`.

    This is the Python 2.x counterpart to `__bool__()` above.

    Raises:
      `TypeError`.
    """
    raise TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed. "
                    "Use `if t is not None:` instead of `if t:` to test if a "
                    "tensor is defined, and use TensorFlow ops such as "
                    "tf.cond to execute subgraphs conditioned on the value of "
                    "a tensor.")

  def eval(self, feed_dict=None, session=None):
    """Evaluates this tensor in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.

    *N.B.* Before invoking `Tensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values.
        See @{tf.Session.run} for a
        description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this tensor. If
        none, the default session will be used.

    Returns:
      A numpy array corresponding to the value of this tensor.

    """
    return _eval_using_default_session(self, feed_dict, self.graph, session)

