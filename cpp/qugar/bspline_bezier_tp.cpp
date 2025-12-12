#include <qugar/bspline_bezier_tp.hpp>
#include <algoim/bezier.hpp>
#include <algoim/xarray.hpp>
#include <memory>
#include <stdexcept>

namespace qugar::impl {

template<int dim, int range> 
BSplineBezierTP<dim, range>::BSplineBezierTP(std::shared_ptr<algoim::bspline::BSplineTP<dim, range>> bspline_tp)
{
  std::vector<std::shared_ptr<algoim::bezier::BezierTP<dim, range>>> algoim_beziers = bspline_tp->splitIntoBeziers();
  beziers_.reserve(algoim_beziers.size());

  for (const auto &abzr : algoim_beziers)
  {
    TensorSizeTP<dim> order(abzr->order);
    
    // The coefficients in our implementation are in the reverse order compared to algoim's.
    beziers_.emplace_back(std::make_shared<BezierTP<dim, range>>(order, this->transpose(abzr->getXarray())));
  }

  for (int i = 0; i < dim; ++i)
    knots_[i] = std::make_shared<const algoim::bspline::Knots>(bspline_tp->getKnots(i));
}

template<int dim, int range>
BSplineBezierTP<dim, range>::Value<real> BSplineBezierTP<dim, range>::operator()(const Point<dim> &point) const
{
  auto multi_index = get_knot_multi_index(point);
  auto bezier_index = get_bezier_index(multi_index);
  auto local_coords = get_local_coordinates(point, multi_index);
  return (*beziers_[bezier_index])(local_coords);
}

template<int dim, int range>
BSplineBezierTP<dim, range>::Value<::algoim::Interval<dim>> BSplineBezierTP<dim, range>::operator()(const Point<dim, Interval<dim>> &point) const
{
  auto multi_index = get_knot_multi_index(point);
  auto bezier_index = get_bezier_index(multi_index);
  auto local_coords = get_local_coordinates(point, multi_index);
  return (*beziers_[bezier_index])(local_coords);
}

template<int dim, int range>
BSplineBezierTP<dim, range>::Gradient<real> BSplineBezierTP<dim, range>::grad(const Point<dim> &point) const
{
  auto multi_index = get_knot_multi_index(point);
  auto bezier_index = get_bezier_index(multi_index);
  auto local_coords = get_local_coordinates(point, multi_index);
  
  Gradient<real> grad = beziers_[bezier_index]->grad(local_coords);
  
  if (range != 1)
    throw std::runtime_error("Gradient rescaling only implemented for range == 1.");

  // Rescale the gradient according to the knot span sizes 
  real knot_start = 0;
  real knot_end = 0;
  real span_size = 0;
  for (int i = 0; i < dim; ++i)
  {
    knot_start = knots_[i]->getUnique()[multi_index[i]];
    knot_end = knots_[i]->getUnique()[multi_index[i] + 1];
    span_size = knot_end - knot_start;
    grad(i) /= span_size;
  }

  return grad;
}

template<int dim, int range>
BSplineBezierTP<dim, range>::Gradient<::algoim::Interval<dim>> BSplineBezierTP<dim, range>::grad(const Point<dim, Interval<dim>> &point) const
{
  throw std::runtime_error("Gradient for Interval points not implemented yet.");
}

template<int dim, int range>
BSplineBezierTP<dim, range>::Hessian<real> BSplineBezierTP<dim, range>::hessian(const Point<dim> &point) const
{
  auto multi_index = get_knot_multi_index(point);
  auto bezier_index = get_bezier_index(multi_index);
  auto local_coords = get_local_coordinates(point, multi_index);
  return beziers_[bezier_index]->hessian(local_coords);
}

template<int dim, int range>
std::shared_ptr<BezierTP<dim, range>> BSplineBezierTP<dim, range>::get_bezier(const Point<dim> &point) const
{
  auto multi_index = get_knot_multi_index(point);
  auto bezier_index = get_bezier_index(multi_index);
  return beziers_[bezier_index];
}

template<int dim, int range>
std::array<int, dim> BSplineBezierTP<dim, range>::get_knot_multi_index(const Point<dim> &point) const
{
  std::array<int, dim> multi_index{};
  for (int i = 0; i < dim; ++i)
    multi_index[i] = knots_[i]->getLastUniqueKnotSmallerOrEqual(point(i));
  return multi_index;
}

template<int dim, int range>
std::array<int, dim> BSplineBezierTP<dim, range>::get_knot_multi_index(const Point<dim, Interval<dim>> &point) const
{
  std::array<int, dim> multi_index{};
  for (int i = 0; i < dim; ++i)
    multi_index[i] = knots_[i]->getLastUniqueKnotSmallerOrEqual(point(i).alpha);
  return multi_index;
}

template<int dim, int range>
int BSplineBezierTP<dim, range>::get_bezier_index(const std::array<int, dim> & multi_index) const
{
  int bezier_index = 0;
  int stride = 1;

  for (int i = dim-1; i >= 0; --i)
  {
    bezier_index += multi_index[i] * stride;
    stride *= (knots_[i]->getNumElements());
  }
    
  return bezier_index;
}

template<int dim, int range>
Point<dim> BSplineBezierTP<dim, range>::get_local_coordinates(const Point<dim> &point, const std::array<int, dim> & multi_index) const
{
  Point<dim> local_coords;
  real knot_start = 0;
  real knot_end = 0;
  for (int i = 0; i < dim; ++i)
  {
    knot_start = knots_[i]->getUnique()[multi_index[i]];
    knot_end = knots_[i]->getUnique()[multi_index[i] + 1];
    local_coords(i) = (point(i) - knot_start) / (knot_end - knot_start);
  }
  return local_coords;
}

template<int dim, int range>
std::vector<typename BSplineBezierTP<dim, range>::CoefsType> BSplineBezierTP<dim, range>::transpose(const ::algoim::xarray<CoefsType, dim> &tensor_rm)
{
  std::vector<CoefsType> tensor_cm;
  tensor_cm.resize(tensor_rm.size());

  auto ext = tensor_rm.ext();

  std::array<int, dim> multi_index{};
  for (int idx = 0; idx < tensor_rm.size(); ++idx)
  {
    // Convert linear index to multi-index (row-major)
    int remainder = idx;
    for (int i = dim - 1; i >= 0; --i)
    {
      multi_index[i] = remainder % ext(i);
      remainder /= ext(i);
    }

    // Convert multi-index to linear index (column-major)
    int cm_index = 0;
    int stride = 1;
    for (int i = 0; i < dim; ++i)
    {
      cm_index += multi_index[i] * stride;
      stride *= ext(i);
    }

    tensor_cm[cm_index] = tensor_rm[idx];
  }

  return tensor_cm;
}



template<int dim, int range>
Point<dim, ::algoim::Interval<dim>> BSplineBezierTP<dim, range>::get_local_coordinates(const Point<dim, Interval<dim>> &point, const std::array<int, dim> & multi_index) const
{
  throw std::runtime_error("get_local_coordinates for Interval points not implemented yet.");
  return Point<dim, Interval<dim>>{};
}

// Explicit template instantiations
template class BSplineBezierTP<1, 1>;
template class BSplineBezierTP<2, 1>;
template class BSplineBezierTP<3, 1>;

template class BSplineBezierTP<1, 3>;
template class BSplineBezierTP<2, 3>;
template class BSplineBezierTP<3, 3>;

} // namespace qugar::impl 
