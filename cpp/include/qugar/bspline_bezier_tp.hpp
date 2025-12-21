#include "domain_function.hpp"
#include <memory>
#include <algoim/bspline.hpp>
#include <qugar/bezier_tp.hpp>


namespace qugar::impl {

template<int dim = 3, int range = 1> 
/* !@brief dim-dimensional tensor-product B-Spline function implemented via
  * a collection of Bézier tensor-products.
  * !tparam dim Dimension of the parametric domain.
  * !tparam range Dimension of the image.
  */
class BSplineBezierTP: public DomainFunc<dim, range>
{

  // TODO: Check the position for the docstring.

public:
  // TODO: This was for PolynomialTP. Check if it is right here.
  //! Parent type.
  using Parent = DomainFunc<dim, range>;

  //! Value type.
  template<typename T> using Value = typename Parent::template Value<T>;

  //! Gradient type.
  template<typename T> using Gradient = typename Parent::template Gradient<T>;

  //! Hessian type.
  template<typename T> using Hessian = typename Parent::template Hessian<T>;

  //! @brief Algoim's interval alias.
  template<int N> using Interval = ::algoim::Interval<N>;

  //! TODO: Check if this is right...
  //! TODO: We always use real here...
  //! Coefs type.
  using CoefsType = std::conditional_t<range == 1, real, ::algoim::uvector<real, range>>;

  //! @brief There's no default BSplineBezierTP.
  BSplineBezierTP() = delete;

  //! @brief Constructor from an algoim's BSplineBezierTP.
  //!
  //! @param bspline Algoim's BSplineTP.
  explicit BSplineBezierTP(std::shared_ptr<algoim::bspline::BSplineTP<dim, range>> bspline_tp);

  // TODO: I don't really get what's going on here. Isn't DomainFunc::operator() abstract?
  //! @brief Inherits the function call operator from the base class.
  //!
  //! Allows instances of this class to be called as functions, utilizing the
  //! (non-virtual) operator() implementation from `DomainFunc<dim, range>` for `Point`
  //! instances.
  using DomainFunc<dim, range>::operator();

  //! @brief Inherits the function grad method from the base class.
  //!
  //! Allows instances of this class to be called as functions, utilizing the
  //! (non-virtual) grad implementation from `DomainFunc<dim, range>` for `Point`
  //! instances.
  using DomainFunc<dim, range>::grad;

private:
  std::vector<std::shared_ptr<BezierTP<dim, range>>> beziers_;
  std::array<std::shared_ptr<const algoim::bspline::Knots>, dim> knots_;

public:
  //! @brief Evaluator operator.
  //!
  //! @param point Point at which the function is evaluated.
  //! @return Function value at @p point.
  [[nodiscard]] virtual Value<real> operator()(const Point<dim> &point) const final;

  //! @brief Evaluator operator.
  //!
  //! @param point Point at which the function is evaluated.
  //! @return Function value at @p point.
  [[nodiscard]] virtual Value<Interval<dim>> operator()(const Point<dim, Interval<dim>> &point) const final;

  //! @brief Gradient evaluator operator.
  //!
  //! @param point Point at which the function's gradient is evaluated.
  //! @return Function gradient at @p point.
  [[nodiscard]] virtual Gradient<real> grad(const Point<dim> &point) const final;

  //! @brief Gradient evaluator operator.
  //!
  //! @param point Point at which the function's gradient is evaluated.
  //! @return Function gradient at @p point.
  [[nodiscard]] virtual Gradient<Interval<dim>> grad(const Point<dim, Interval<dim>> &point) const final;

  //! @brief Hessian evaluator operator.
  //!
  //! @param point Point at which the function's hessian is evaluated.
  //! @return Function hessian at @p point.
  [[nodiscard]] Hessian<real> virtual hessian(const Point<dim> &point) const final;

  //! @brief Getter for the underlying Bézier functions based on a given point.
  //!
  //! @param point Point in the parametric domain.
  //! @return Shared pointer to the corresponding Bézier function.
  [[nodiscard]] std::shared_ptr<BezierTP<dim, range>> get_bezier(const Point<dim> &point) const;

  //! @brief Calculate the knot multi-index corresponding to a given point.
  //!
  //! @param point Point in the parametric domain.
  //! @return Knot multi-index as an array of integers.
  [[nodiscard]] std::array<int, dim> get_knot_multi_index(const Point<dim> &point) const;

  //! @brief Calculate the knot multi-index corresponding to a given interval point.
  //!
  //! @param point Point in the parametric domain.
  //! @return Knot multi-index as an array of integers.
  [[nodiscard]] std::array<int, dim> get_knot_multi_index(const Point<dim, Interval<dim>> &point) const;

  //! @brief Calculate the index of the Bézier corresponding to a given point.
  //! //! @param point Point in the parametric domain.
  //! @return Index of the corresponding Bézier.
  [[nodiscard]] int get_bezier_index(const std::array<int, dim> & multi_index) const;

  //! @brief Calculate the local coordinates of a point within its knot span.
  //!
  //! @param point Point in the parametric domain.
  //! @return Local coordinates as a Point object.
  [[nodiscard]] Point<dim> get_local_coordinates(const Point<dim> &point, const std::array<int, dim> & multi_index) const;

  //! @brief Calculate the local coordinates of an interval point within its knot span.
  //! @param point Point in the parametric domain.
  //! @return Local coordinates as a Point object.
  [[nodiscard]] Point<dim, Interval<dim>> get_local_coordinates(const Point<dim, Interval<dim>> &point, const std::array<int, dim> & multi_index) const;

  //! @brief A function to turn the coefficient flattened tensor from row-major
  //! order used in algoim to column-major order used in qugar.
  //!
  //! @param tensor_rm Coefficient tensor in row-major order.
  //! @return Coefficient tensor in column-major order.
  static std::vector<CoefsType> transpose(const ::algoim::xarray<CoefsType, dim> &tensor_rm);

  // TODO: These methods should be moved out of qugar completely.
  //! @brief Factory method to create a BSplineBezierTP from B-Spline parameters.
  //! It is assumed that the knot vectors are open, uniform and with maximum regularity.
  //! Range is assumed to be 1.
  //!
  //! @param knots_min Minimum knot values for each dimension.
  //! @param knots_max Maximum knot values for each dimension.
  //! @param num_spans Number of spans for each dimension.
  //! @param order Order of the B-Spline for each dimension.
  //! @param coefficients Coefficients of the B-Spline tensor-product function,
  //! ordered such that dimension N-1 is inner-most.
  static std::shared_ptr<BSplineBezierTP<dim>> form_bspline(const std::array<real, dim> &knots_min,
                                                            const std::array<real, dim> &knots_max,
                                                            const std::array<int, dim>  &num_spans,
                                                            const std::array<int, dim>  &order,
                                                            const std::vector<real>     &coefficients);

  //! @brief Factory method to create a BSplineBezierTP from B-Spline parameters.
  //! Range is assumed to be 1.
  //!
  /// @param knots Knot vectors for each dimension.
  //! @param order Order of the B-Spline for each dimension. This means degree + 1.
  //! @param coefficients Coefficients of the B-Spline tensor-product function,
  //! ordered such that dimension N-1 is inner-most.
  static std::shared_ptr<BSplineBezierTP<dim>> form_bspline(const std::array<std::vector<real>, dim> &knots,
                                                            const std::array<int, dim>               &order,
                                                            const std::vector<real>                  &coefficients);
};


//! @brief Checks if a given DomainFunc object is of type BSplineBezierTP. 
//!
//! This function uses dynamic_cast to determine if the provided DomainFunc
//! object can be cast to a BSplineBezierTP object. If the cast is successful, 
//! the function returns true, indicating that the object is of type BezierTP.
//! Otherwise, it returns false.
//!
//! @tparam dim Dimension of the parametric domain.
//! @tparam range Dimension of the image.
//! @param func DomainFunc object to check.
//! @return `true` if the object is of type BSplineBezierTP; `false` otherwise.
template<int dim, int range>
static bool is_bspline_bezier(const DomainFunc<dim, range> &func)
{
  const auto *bspline_bezier = dynamic_cast<const BSplineBezierTP<dim, range> *>(&func);
  return bspline_bezier != nullptr;
}

} // namespace qugar::impl
