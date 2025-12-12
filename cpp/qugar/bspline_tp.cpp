#include <qugar/bspline_tp.hpp>
#include <memory>

namespace qugar::impl {

template<int dim, int range>
BSplineTP<dim, range>::BSplineTP(std::shared_ptr<algoim::bspline::BSplineTP<dim, range>> bspline)
  : bspline_(std::move(bspline)) { }

template<int dim, int range>
BSplineTP<dim, range>::Value<real> BSplineTP<dim, range>::operator()(const Point<dim> &point) const
{
  return bspline_->operator()(point);
}

template<int dim, int range>
BSplineTP<dim, range>::Value<::algoim::Interval<dim>> BSplineTP<dim, range>::operator()(const Point<dim, Interval<dim>> &point) const
{
  return bspline_->operator()(point);
}

template<int dim, int range>
BSplineTP<dim, range>::Gradient<real> BSplineTP<dim, range>::grad(const Point<dim> &point) const
{
  return bspline_->grad(point);
}

template<int dim, int range>
BSplineTP<dim, range>::Gradient<::algoim::Interval<dim>> BSplineTP<dim, range>::grad(const Point<dim, Interval<dim>> &point) const
{
  return bspline_->grad(point);
}

template<int dim, int range>
BSplineTP<dim, range>::Hessian<real> BSplineTP<dim, range>::hessian(const Point<dim> &point) const
{
  return bspline_->hessian(point);
}


// Instantiations
template class qugar::impl::BSplineTP<3, 1>;

} // namespace qugar::impl
