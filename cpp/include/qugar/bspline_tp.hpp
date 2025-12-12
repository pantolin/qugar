#include "domain_function.hpp"
#include <memory>
#include <algoim/bspline.hpp>


namespace qugar::impl {

template<int dim = 3, int range = 1> 
class BSplineTP: public DomainFunc<dim, range>
{
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

  //! @brief There's no default BSplineTP.
  BSplineTP() = delete;

  //! @brief Constructor from an algoim's BSplineTP.
  //!
  //! @param bspline Algoim's BSplineTP.
  explicit BSplineTP(std::shared_ptr<algoim::bspline::BSplineTP<dim, range>> bspline);

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
  std::shared_ptr<algoim::bspline::BSplineTP<dim, range>> bspline_;

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

};


//! @brief Checks if a given DomainFunc object is of type BezierTP.
//!
//! This function uses dynamic_cast to determine if the provided DomainFunc
//! object can be cast to a BezierTP object. If the cast is successful, the
//! function returns true, indicating that the object is of type BezierTP.
//! Otherwise, it returns false.
//!
//! @tparam dim The dimension of the BezierTP object.
//! @tparam range The range of the BezierTP object.
//! @param func The DomainFunc object to be checked.
//! @return true if the object is of type BezierTP, false otherwise.
template<int dim, int range> static bool is_bspline(const DomainFunc<dim, range> &func)
{
  const auto *bzr = dynamic_cast<const BSplineTP<dim, range> *>(&func);
  return bzr != nullptr;
}

} // namespace qugar::impl
