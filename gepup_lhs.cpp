// 编译默认的是debug模式，正式算的话要make release之后再make
// mpirun -np 并行的进程数 ./gepup 全局加密次数
// mpirun -np 4 ./gepup 4

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <chrono>
#include <filesystem>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>

using namespace dealii;
std::ifstream file("parameters.txt");
std::string name;
double value;

const unsigned int finite_element_degree = 4; // 单元阶数
const unsigned int dimension = 2;             // 问题维数

const double PI = 4.0 * (atan(1.0));
const double vis = 1.0 / 1000;  // 粘性系数
const double gamma_coef = 0.25; // 时间积分的参数，不能改

unsigned int timestep_number = 0; // 初始时间步，0步
double time_end = 1;              // 终止时间
double time_step = 0.001;         // 时间步长
double parameter = 1.0;                // 默认速度值

struct ParameterReader {
    ParameterReader() {
        std::ifstream file("parameters.txt");
        std::string name;
        double value;

        while (file >> name >> value)
        {
            if (name == "parameter")
                parameter = value;
        }
    }
} parameterReader; // 创建一个ParameterReader对象，这将执行构造函数中的代码

template <int dim>
class MultiComponentFunction : public Function<dim>
{
public:
    MultiComponentFunction(const double initial_time = 0)
        : Function<dim>(1, initial_time), comp(0)
    {
    }
    void set_component(const unsigned int d)
    {
        comp = d;
    }

protected:
    unsigned int comp; // 分量
};
/*
// 二维有解析解的非齐次边界条件
// 速度值
template <int dim>
class VelocityValue : public MultiComponentFunction<dim>
{
public:
    VelocityValue(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {
    }
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double x = p(0);
        double y = p(1);
        double t = this->get_time();
        if (this->comp == 0)
            return sinh(x) * sinh(x) * sinh(2 * y) * cos(PI * t);
        else
            return -sinh(y) * sinh(y) * sinh(2 * x) * cos(PI * t); // lhs论文中少写一个负号，此处正确
    }
    // 将值设定为函数指定comp在点处的值
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityValue<dim>::value(points[i]);
    }
};

// 速度边界值
template <int dim>
class VelocityBoundary : public MultiComponentFunction<dim>
{
public:
    VelocityBoundary(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {
    }
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double x = p(0);
        double y = p(1);
        double t = this->get_time();
        if (this->comp == 0)
            return sinh(x) * sinh(x) * sinh(2 * y) * cos(PI * t);
        else
            return -sinh(y) * sinh(y) * sinh(2 * x) * cos(PI * t);
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityBoundary<dim>::value(points[i]);
    }
};

// u_t
template <int dim>
class VelocityBoundary_t : public MultiComponentFunction<dim>
{
public:
    VelocityBoundary_t(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {
    }
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double x = p(0);
        double y = p(1);
        double t = this->get_time();
        if (this->comp == 0)
            return -sinh(x) * sinh(x) * sinh(2 * y) * PI * sin(PI * t);
        else
            return sinh(y) * sinh(y) * sinh(2 * x) * PI * sin(PI * t);
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityBoundary_t<dim>::value(points[i]);
    }
};

// 外力项g，散度:gx_x+gy_y
template <int dim>
class BodyForce : public MultiComponentFunction<dim>
{
public:
    BodyForce(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {
    }
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double x = p(0);
        double y = p(1);
        double t = this->get_time();
        if (this->comp == 0)
            return 2 * sinh(2 * y) * sinh(2 * y) * cos(PI * t) * cos(PI * t) * cosh(x) * pow(sinh(x), 3) - cos(PI * t) * sinh(x) * sinh(y) - vis * (2 * sinh(2 * y) * cos(PI * t) * cosh(x) * cosh(x) + 6 * sinh(2 * y) * cos(PI * t) * sinh(x) * sinh(x)) - PI * sinh(2 * y) * sin(PI * t) * sinh(x) * sinh(x) - 2 * cosh(2 * y) * sinh(2 * x) * cos(PI * t) * cos(PI * t) * sinh(x) * sinh(x) * sinh(y) * sinh(y);
        else
            return vis * (2 * sinh(2 * x) * cos(PI * t) * cosh(y) * cosh(y) + 6 * sinh(2 * x) * cos(PI * t) * sinh(y) * sinh(y)) - cos(PI * t) * cosh(x) * cosh(y) + 2 * sinh(2 * x) * sinh(2 * x) * cos(PI * t) * cos(PI * t) * cosh(y) * pow(sinh(y), 3) + PI * sinh(2 * x) * sin(PI * t) * sinh(y) * sinh(y) - 2 * cosh(2 * x) * sinh(2 * y) * cos(PI * t) * cos(PI * t) * sinh(x) * sinh(x) * sinh(y) * sinh(y);
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = BodyForce<dim>::value(points[i]);
    }
    double divergence(const Point<dim> &p, const unsigned int = 0)
    {
        double x = p(0);
        double y = p(1);
        double t = this->get_time();
        return 2 * sinh(2 * y) * sinh(2 * y) * cos(PI * t) * cos(PI * t) * pow(sinh(x), 4) - cos(PI * t) * cosh(x) * sinh(y) + 6 * sinh(2 * y) * sinh(2 * y) * cos(PI * t) * cos(PI * t) * cosh(x) * cosh(x) * sinh(x) * sinh(x) - 4 * cosh(2 * x) * cosh(2 * y) * cos(PI * t) * cos(PI * t) * sinh(x) * sinh(x) * sinh(y) * sinh(y) - 2 * PI * sinh(2 * y) * sin(PI * t) * cosh(x) * sinh(x) - 16 * vis * sinh(2 * y) * cos(PI * t) * cosh(x) * sinh(x) - 4 * cosh(2 * y) * sinh(2 * x) * cos(PI * t) * cos(PI * t) * cosh(x) * sinh(x) * sinh(y) * sinh(y) + 2 * sinh(2 * x) * sinh(2 * x) * cos(PI * t) * cos(PI * t) * pow(sinh(y), 4) - cos(PI * t) * cosh(x) * sinh(y) + 6 * sinh(2 * x) * sinh(2 * x) * cos(PI * t) * cos(PI * t) * cosh(y) * cosh(y) * sinh(y) * sinh(y) - 4 * cosh(2 * x) * cosh(2 * y) * cos(PI * t) * cos(PI * t) * sinh(x) * sinh(x) * sinh(y) * sinh(y) + 2 * PI * sinh(2 * x) * sin(PI * t) * cosh(y) * sinh(y) + 16 * vis * sinh(2 * x) * cos(PI * t) * cosh(y) * sinh(y) - 4 * cosh(2 * x) * sinh(2 * y) * cos(PI * t) * cos(PI * t) * cosh(y) * sinh(x) * sinh(x) * sinh(y);
    }
    void divergence_list(const std::vector<Point<dim>> &points, std::vector<double> &divergencies, const unsigned int = 0)
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            divergencies[i] = BodyForce<dim>::divergence(points[i]);
    }
};
*/

//二维方腔流
template <int dim>
class VelocityValue : public MultiComponentFunction<dim>
{
public:
    VelocityValue(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        (void) p;
        double epsilon = 1e-12;
        if (this->comp == 0 && fabs(1 - p(1)) < epsilon)
            return parameter;
        else
            return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityValue<dim>::value(points[i]);
    }
};

template <int dim>
class VelocityBoundary : public MultiComponentFunction<dim>
{
public:
    VelocityBoundary(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        (void) p;
        double epsilon = 1e-12;
        if (this->comp == 0 && fabs(1 - p(1)) < epsilon)
            return parameter;
        else
            return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityBoundary<dim>::value(points[i]);
    }
};

template <int dim>
class VelocityBoundary_t : public MultiComponentFunction<dim>
{
public:
    VelocityBoundary_t(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        (void) p;
        return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityBoundary_t<dim>::value(points[i]);
    }
};

template <int dim>
class BodyForce : public MultiComponentFunction<dim>
{
public:
    BodyForce(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        (void) p;
        return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = BodyForce<dim>::value(points[i]);
    }
    double divergence(const Point<dim> &p, const unsigned int = 0)
    {
        (void) p;
        return 0;
    }
    void divergence_list(const std::vector<Point<dim>> &points, std::vector<double> &divergencies, const unsigned int = 0)
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            divergencies[i] = BodyForce<dim>::divergence(points[i]);
    }
};

/*
//二维圆柱绕流
template <int dim>
class VelocityValue : public MultiComponentFunction<dim>
{
public:
    VelocityValue(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double epsilon = 1e-12;
        double t = this->get_time();
        if (this->comp == 0 && (fabs(p(0)) < epsilon || fabs(2.2 - p(0)) < epsilon))
            return 6.0 / (0.41 * 0.41) * sin(PI * t / 8.0) * p(1) * (0.41 - p(1));
        else
            return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityValue<dim>::value(points[i]);
    }
};

template <int dim>
class VelocityBoundary : public MultiComponentFunction<dim>
{
public:
    VelocityBoundary(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double epsilon = 1e-12;
        double t = this->get_time();
        if (this->comp == 0 && (fabs(p(0)) < epsilon || fabs(2.2 - p(0)) < epsilon))
            return 6.0 / (0.41 * 0.41) * sin(PI * t / 8.0) * p(1) * (0.41 - p(1));
        else
            return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityBoundary<dim>::value(points[i]);
    }
};

template <int dim>
class VelocityBoundary_t : public MultiComponentFunction<dim>
{
public:
    VelocityBoundary_t(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double epsilon = 1e-12;
        double t = this->get_time();
        if (this->comp == 0 && (fabs(p(0)) < epsilon || fabs(2.2 - p(0)) < epsilon))
            return PI / 8.0 * 6.0 / (0.41 * 0.41) * cos(PI * t / 8.0) * p(1) * (0.41 - p(1));
        else
            return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityBoundary_t<dim>::value(points[i]);
    }
};

template <int dim>
class BodyForce : public MultiComponentFunction<dim>
{
public:
    BodyForce(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        (void) p;
        return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = BodyForce<dim>::value(points[i]);
    }
    double divergence(const Point<dim> &p, const unsigned int = 0)
    {
        (void) p;
        return 0;
    }
    void divergence_list(const std::vector<Point<dim>> &points, std::vector<double> &divergencies, const unsigned int = 0)
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            divergencies[i] = BodyForce<dim>::divergence(points[i]);
    }
};
*/

/*
//三维流体绕过一个球体，圆柱直径为0.41，球的直径为0.1
const double cylinder_center = 0.005 / 1.41421356237309504880168872420969807856967187537694807317667973799;
double inner_cube_half_length = 0.058578643762690500718104402722019585780799388885498046875;
template <int dim>
class VelocityValue : public MultiComponentFunction<dim>
{
public:
    VelocityValue(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double epsilon = 1e-12;
        //double t = this->get_time();
        if (this->comp == 0 && (fabs(p(0) + 20 * inner_cube_half_length) < epsilon || fabs(p(0) - 20 * inner_cube_half_length) < epsilon))
            return 1.5 * (1 - (p(1) - cylinder_center) * (p(1) - cylinder_center) / 0.205 / 0.205 - (p(2) - cylinder_center) * (p(2) - cylinder_center) / 0.205 / 0.205);
        else
            return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityValue<dim>::value(points[i]);
    }
};

template <int dim>
class VelocityBoundary : public MultiComponentFunction<dim>
{
public:
    VelocityBoundary(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        double epsilon = 1e-12;
        //double t = this->get_time();
        if (this->comp == 0 && (fabs(p(0) + 20 * inner_cube_half_length) < epsilon || fabs(p(0) - 20 * inner_cube_half_length) < epsilon))
            return 1.5 * (1 - (p(1) - cylinder_center) * (p(1) - cylinder_center) / 0.205 / 0.205 - (p(2) - cylinder_center) * (p(2) - cylinder_center) / 0.205 / 0.205);
        else
            return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityBoundary<dim>::value(points[i]);
    }
};

template <int dim>
class VelocityBoundary_t : public MultiComponentFunction<dim>
{
public:
    VelocityBoundary_t(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        (void) p;
        return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = VelocityBoundary_t<dim>::value(points[i]);
    }
};

template <int dim>
class BodyForce : public MultiComponentFunction<dim>
{
public:
    BodyForce(const double initial_time = 0)
        : MultiComponentFunction<dim>(initial_time)
    {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        (void) p;
        return 0;
    }
    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<double> &values, const unsigned int = 0) const override
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            values[i] = BodyForce<dim>::value(points[i]);
    }
    double divergence(const Point<dim> &p, const unsigned int = 0)
    {
        (void) p;
        return 0;
    }
    void divergence_list(const std::vector<Point<dim>> &points, std::vector<double> &divergencies, const unsigned int = 0)
    {
        const unsigned int n_points = points.size();
        for (unsigned int i = 0; i < n_points; ++i)
            divergencies[i] = BodyForce<dim>::divergence(points[i]);
    }
};
*/

// matrix-free方法下的矩阵M+c*A如何计算
template <int dim, int fe_degree, typename number>
class HelmholtzOperator : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:
    using value_type = number;

    HelmholtzOperator();
    // void clear() override;
    void evaluate_coefficient();
    virtual void compute_diagonal() override;

private:
    virtual void apply_add(LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src) const override;
    void local_apply(const MatrixFree<dim, number> &data, LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src, const std::pair<unsigned int, unsigned int> &cell_range) const;
    void local_compute_diagonal(const MatrixFree<dim, number> &data, LinearAlgebra::distributed::Vector<number> &dst, const unsigned int &dummy, const std::pair<unsigned int, unsigned int> &cell_range) const;

    double coefficient = time_step * vis * gamma_coef;
};

template <int dim, int fe_degree, typename number>
HelmholtzOperator<dim, fe_degree, number>::HelmholtzOperator() : MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>()
{
}

template <int dim, int fe_degree, typename number>
void HelmholtzOperator<dim, fe_degree, number>::evaluate_coefficient()
{
    coefficient = time_step * vis * gamma_coef;
}

template <int dim, int fe_degree, typename number>
void HelmholtzOperator<dim, fe_degree, number>::local_apply(const MatrixFree<dim, number> &data, LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src, const std::pair<unsigned int, unsigned int> &cell_range) const
{
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
            phi.submit_value(phi.get_value(q), q);
            phi.submit_gradient(coefficient * phi.get_gradient(q), q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
}

template <int dim, int fe_degree, typename number>
void HelmholtzOperator<dim, fe_degree, number>::apply_add(LinearAlgebra::distributed::Vector<number> &dst, const LinearAlgebra::distributed::Vector<number> &src) const
{
    this->data->cell_loop(&HelmholtzOperator::local_apply, this, dst, src);
}

template <int dim, int fe_degree, typename number>
void HelmholtzOperator<dim, fe_degree, number>::compute_diagonal()
{
    this->inverse_diagonal_entries.reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
    unsigned int dummy = 0;
    this->data->cell_loop(&HelmholtzOperator::local_compute_diagonal, this, inverse_diagonal, dummy);

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
        inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
}

template <int dim, int fe_degree, typename number>
void HelmholtzOperator<dim, fe_degree, number>::local_compute_diagonal(const MatrixFree<dim, number> &data, LinearAlgebra::distributed::Vector<number> &dst, const unsigned int &, const std::pair<unsigned int, unsigned int> &cell_range) const
{
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

    AlignedVector<VectorizedArray<number>> diagonal(phi.dofs_per_cell);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
        phi.reinit(cell);
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
        {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                phi.submit_dof_value(VectorizedArray<number>(), j);
            phi.submit_dof_value(make_vectorized_array<number>(1.), i);

            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
                phi.submit_value(phi.get_value(q), q);
                phi.submit_gradient(coefficient * phi.get_gradient(q), q);
            }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            diagonal[i] = phi.get_dof_value(i);
        }
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global(dst);
    }
}

template <int dim>
class GePUP
{
public:
    GePUP(unsigned int n_global_refine);
    void run();
    void run_adaptive();

private:
    void make_grid();
    void setup_dofs();
    void initialize_u();
    void compute_old_w_integral();
    void compute_q_and_get_minus_u_dot_grad_u_integral(double current_time, unsigned int step);
    void compute_ex_integral_and_laplacian_w_integral(double current_time, unsigned int step);
    void compute_w(double current_time, unsigned int step);
    void compute_phi(double current_time);
    void compute_u(double current_time);
    void compute_w_star(double current_time);
    void run_once();
    void refine_mesh(const unsigned int min_grid_level, const unsigned int max_grid_level);
    double get_cfl_number();
    void output_image(std::string file_name, unsigned int output_count, double current_time);
    void print_error();
    void compute_vorticity();
    void compute_vorticity_3d();

    MPI_Comm mpi_communicator;
    const unsigned int this_mpi_process;
    ConditionalOStream pcout;
    TimerOutput computing_timer;
    unsigned int n_global_refine;

    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    FE_Q<dim> fe;
    MappingQ<dim> mapping;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> hanging_node_constraints;
    AffineConstraints<double> zero_boundary_and_hanging_node_constraints;
    AffineConstraints<double> one_dof_zero_and_hanging_node_constraints;

    LinearAlgebra::distributed::Vector<double> u[dim];
    LinearAlgebra::distributed::Vector<double> u_rhs[dim];
    LinearAlgebra::distributed::Vector<double> w[dim];
    LinearAlgebra::distributed::Vector<double> w_rhs[dim];
    LinearAlgebra::distributed::Vector<double> q;
    LinearAlgebra::distributed::Vector<double> q_phi_rhs;
    LinearAlgebra::distributed::Vector<double> phi;
    LinearAlgebra::distributed::Vector<double> old_w_integral[dim];
    std::array<LinearAlgebra::distributed::Vector<double>, 6> ex_integral[dim];
    std::array<LinearAlgebra::distributed::Vector<double>, 5> laplacian_w_integral[dim];
    LinearAlgebra::distributed::Vector<double> tmp_vec;
    LinearAlgebra::distributed::Vector<double> tmp_rhs_vec;

    LinearAlgebra::distributed::Vector<double> p;

    LinearAlgebra::distributed::Vector<double> vorticity;
    LinearAlgebra::distributed::Vector<double> vorticity_rhs;

    using MassMatrixType = MatrixFreeOperators::MassOperator<dim, finite_element_degree, finite_element_degree + 1, 1, LinearAlgebra::distributed::Vector<double>>;
    using HelmholtzMatrixType = HelmholtzOperator<dim, finite_element_degree, double>;
    using LaplaceMatrixType = MatrixFreeOperators::LaplaceOperator<dim, finite_element_degree, finite_element_degree + 1, 1, LinearAlgebra::distributed::Vector<double>>;

    using MassLevelMatrixType = MatrixFreeOperators::MassOperator<dim, finite_element_degree, finite_element_degree + 1, 1, LinearAlgebra::distributed::Vector<float>>;
    using HelmholtzLevelMatrixType = HelmholtzOperator<dim, finite_element_degree, float>;
    using LaplaceLevelMatrixType = MatrixFreeOperators::LaplaceOperator<dim, finite_element_degree, finite_element_degree + 1, 1, LinearAlgebra::distributed::Vector<float>>;

    MassMatrixType u_matrix;
    HelmholtzMatrixType w_matrix;
    LaplaceMatrixType q_phi_matrix;

    MGConstrainedDoFs u_w_mg_constrained_dofs;
    MGConstrainedDoFs q_phi_mg_constrained_dofs;

    MGLevelObject<MassLevelMatrixType> u_mg_matrices;
    MGLevelObject<HelmholtzLevelMatrixType> w_mg_matrices;
    MGLevelObject<LaplaceLevelMatrixType> q_phi_mg_matrices;

    MassMatrixType vorticity_matrix;
    MGConstrainedDoFs vorticity_mg_constrained_dofs;
    MGLevelObject<MassLevelMatrixType> vorticity_mg_matrices;

    VelocityValue<dim> velocity_value;
    VelocityBoundary<dim> velocity_boundary;
    VelocityBoundary_t<dim> velocity_boundary_t;
    BodyForce<dim> body_force;

    std::array<double, 6> c;
    std::array<double, 6> b;
    std::array<std::array<double, 6>, 6> a_ex;
    std::array<std::array<double, 6>, 6> a_im;

    double time;
};

template <int dim>
GePUP<dim>::GePUP(unsigned int n_global_refine)
    : mpi_communicator(MPI_COMM_WORLD)
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
    , n_global_refine(n_global_refine)
    , triangulation(mpi_communicator, Triangulation<dim>::limit_level_difference_at_vertices, parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
    , dof_handler(triangulation)
    , fe(finite_element_degree)
    , mapping(finite_element_degree)
    , time(0.)
{
    c[0] = 0.0;
    c[1] = 0.5;
    c[2] = 0.332;
    c[3] = 0.62;
    c[4] = 0.85;
    c[5] = 1.0;
    b[0] = 0.15791629516167136;
    b[1] = 0.0;
    b[2] = 0.18675894052400077;
    b[3] = 0.6805652953093346;
    b[4] = -0.27524053099500667;
    b[5] = gamma_coef;
    a_ex[1][0] = 2 * gamma_coef;
    a_ex[2][0] = 0.221776;
    a_ex[2][1] = 0.110224;
    a_ex[3][0] = -0.04884659515311857;
    a_ex[3][1] = -0.17772065232640102;
    a_ex[3][2] = 0.8465672474795197;
    a_ex[4][0] = -0.15541685842491548;
    a_ex[4][1] = -0.3567050098221991;
    a_ex[4][2] = 1.0587258798684427;
    a_ex[4][3] = 0.30339598837867193;
    a_ex[5][0] = 0.2014243506726763;
    a_ex[5][1] = 0.008742057842904185;
    a_ex[5][2] = 0.15993995707168115;
    a_ex[5][3] = 0.4038290605220775;
    a_ex[5][4] = 0.22606457389066084;
    a_ex[5][5] = 0.0;
    a_im[1][0] = gamma_coef;
    a_im[2][0] = 0.137776;
    a_im[2][1] = -0.055776;
    a_im[3][0] = 0.14463686602698217;
    a_im[3][1] = -0.22393190761334475;
    a_im[3][2] = 0.4492950415863626;
    a_im[4][0] = 0.09825878328356477;
    a_im[4][1] = -0.5915442428196704;
    a_im[4][2] = 0.8101210538282996;
    a_im[4][3] = 0.283164405707806;
    a_im[5][0] = b[0];
    a_im[5][1] = b[1];
    a_im[5][2] = b[2];
    a_im[5][3] = b[3];
    a_im[5][4] = b[4];
}

template <int dim>
void GePUP<dim>::make_grid()
{
    TimerOutput::Scope t(computing_timer, "make grid");

    GridGenerator::hyper_cube(triangulation, 0, 1);

    // GridGenerator::channel_with_cylinder(triangulation, 0.03, 1, 0);

    /*
    //三维绕球
    Point<dim> ball_center(-inner_cube_half_length * 15, 0, 0);

    GridGenerator::subdivided_cylinder(triangulation, 20, 0.2, inner_cube_half_length * 20);
    std::set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;

    for (const auto &cell : triangulation.active_cell_iterators())
    {
        const auto center = cell->center();
        if (center.distance(ball_center) < 1e-5)
            cells_to_remove.insert(cell);
    }
    GridGenerator::create_triangulation_with_removed_cells(triangulation, cells_to_remove, triangulation);

    parallel::distributed::Triangulation<dim> triangulation2(mpi_communicator);
    GridGenerator::hyper_shell(triangulation2, ball_center, 0.05, inner_cube_half_length * 1.73205080756887729352744634150587236694280525381038062805580, 6);
    GridGenerator::merge_triangulations(triangulation, triangulation2, triangulation);

    for (const auto &cell : triangulation.active_cell_iterators())
        for (const auto i : cell->vertex_indices())
        {
            Point<dim> &v = cell->vertex(i);
            if (std::abs(v(1) - 0.2 / 1.41421356237309504880168872420969807856967187537694807317667973799) < 1e-5)
                v(1) += 0.01 / 1.41421356237309504880168872420969807856967187537694807317667973799;
            if (std::abs(v(2) - 0.2 / 1.41421356237309504880168872420969807856967187537694807317667973799) < 1e-5)
                v(2) += 0.01 / 1.41421356237309504880168872420969807856967187537694807317667973799;
        }

    triangulation.reset_all_manifolds();
    triangulation.set_all_manifold_ids(0);

    for (const auto &cell : triangulation.cell_iterators())
        for (const auto &face : cell->face_iterators())
        {
            bool face_at_sphere_boundary = true;
            for (const auto v : face->vertex_indices())
                if (fabs(face->vertex(v).distance(ball_center) - 0.05) > 1e-12)
                    face_at_sphere_boundary = false;
            if (face_at_sphere_boundary)
                face->set_all_manifold_ids(1);
        }
    for (const auto &cell : triangulation.cell_iterators())
        for (const auto &face : cell->face_iterators())
        {
            bool face_at_sphere_boundary = true;
            for (const auto v : face->vertex_indices())
                if (fabs(face->vertex(v).distance(ball_center) - inner_cube_half_length * 1.73205080756887729352744634150587236694280525381038062805580) > 1e-12)
                    face_at_sphere_boundary = false;
            if (face_at_sphere_boundary)
                face->set_all_manifold_ids(1);
        }
    triangulation.set_manifold(1, SphericalManifold<dim>(ball_center));

    Tensor<1, dim> cylinder_direction({1, 0, 0});
    Point<dim> point_on_axis(0, 0.005 / 1.41421356237309504880168872420969807856967187537694807317667973799, 0.005 / 1.41421356237309504880168872420969807856967187537694807317667973799);
    CylindricalManifold<dim> cylindrical_manifold(cylinder_direction, point_on_axis);
    for (const auto &cell : triangulation.cell_iterators())
        for (const auto &face : cell->face_iterators())
            if (face->at_boundary())
            {
                const auto center = face->center();
                if (std::fabs(center(1) + 0.141421356237309504880168872420969807856967187537694807317667973799) < 1e-5 || std::fabs(center(2) + 0.141421356237309504880168872420969807856967187537694807317667973799) < 1e-5 || std::fabs(center(1) - 0.1484924240491749813930510981663246639072895050048828125) < 1e-5 || std::fabs(center(2) - 0.1484924240491749813930510981663246639072895050048828125) < 1e-5)
                    face->set_all_manifold_ids(2);
            }
    triangulation.set_manifold(2, cylindrical_manifold);
    */

    triangulation.refine_global(n_global_refine);
    //网格的长度h
    double max_active_cell_diameter = 0.0;
    double max_nonactive_cell_diameter = 0.0;
    double min_active_cell_diameter = std::numeric_limits<double>::max();
    double min_nonactive_cell_diameter = std::numeric_limits<double>::max();
    for (const auto &cell : triangulation.active_cell_iterators()) {
        double cell_diameter = cell->diameter();
        max_active_cell_diameter = std::max(max_active_cell_diameter, cell_diameter);
        min_active_cell_diameter = std::min(min_active_cell_diameter, cell_diameter);
    }

    for (const auto &cell : triangulation.cell_iterators()) {
        if (!cell->is_active()) {
        double cell_diameter = cell->diameter();
        max_nonactive_cell_diameter = std::max(max_nonactive_cell_diameter, cell_diameter);
        min_nonactive_cell_diameter = std::min(min_nonactive_cell_diameter, cell_diameter);
        }
    }
    std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "Maximum active cell diameter: " << max_active_cell_diameter << std::endl;
    std::cout << "Minimum active cell diameter: " << min_active_cell_diameter << std::endl;
    std::cout << "Maximum non-active cell diameter: " << max_nonactive_cell_diameter << std::endl;
    std::cout << "Minimum non-active cell diameter: " << min_nonactive_cell_diameter << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    // GridGenerator::subdivided_hyper_cube(triangulation, n_global_refine, 0, 1);
}

template <int dim>
void GePUP<dim>::setup_dofs()
{
    TimerOutput::Scope t(computing_timer, "setup dofs");

    // 这段好像没什么用，可加可不加
    u_matrix.clear();
    w_matrix.clear();
    q_phi_matrix.clear();
    u_mg_matrices.clear_elements();
    w_mg_matrices.clear_elements();
    q_phi_mg_matrices.clear_elements();

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();
    pcout << "Number of dofs: " << dof_handler.n_dofs() << std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    for (unsigned int d = 0; d < dim; ++d)
    {
        for (unsigned int s = 0; s < 5; ++s)
        {
            ex_integral[d][s].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
            laplacian_w_integral[d][s].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        }
        ex_integral[d][5].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        old_w_integral[d].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    }
    vorticity.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    hanging_node_constraints.clear();
    hanging_node_constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
    hanging_node_constraints.close();

    zero_boundary_and_hanging_node_constraints.clear();
    zero_boundary_and_hanging_node_constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, zero_boundary_and_hanging_node_constraints);
    VectorTools::interpolate_boundary_values(mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), zero_boundary_and_hanging_node_constraints);
    zero_boundary_and_hanging_node_constraints.close();

    one_dof_zero_and_hanging_node_constraints.clear();
    one_dof_zero_and_hanging_node_constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, one_dof_zero_and_hanging_node_constraints);
    if (hanging_node_constraints.can_store_line(0))
        one_dof_zero_and_hanging_node_constraints.add_line(0);
    one_dof_zero_and_hanging_node_constraints.close();

    {
        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_values | update_JxW_values);
        std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(new MatrixFree<dim, double>());
        system_mf_storage->reinit(mapping, dof_handler, zero_boundary_and_hanging_node_constraints, QGauss<1>(fe.degree + 1), additional_data);
        u_matrix.initialize(system_mf_storage);
    }
    for (unsigned int d = 0; d < dim; ++d)
    {
        u_matrix.initialize_dof_vector(u[d]);
        u_matrix.initialize_dof_vector(u_rhs[d]);
    }
    {
        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values);
        std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(new MatrixFree<dim, double>());
        system_mf_storage->reinit(mapping, dof_handler, zero_boundary_and_hanging_node_constraints, QGauss<1>(fe.degree + 1), additional_data);
        w_matrix.initialize(system_mf_storage);
    }
    w_matrix.evaluate_coefficient();
    for (unsigned int d = 0; d < dim; ++d)
    {
        w_matrix.initialize_dof_vector(w[d]);
        w_matrix.initialize_dof_vector(w_rhs[d]);
    }
    {
        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values);
        std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(new MatrixFree<dim, double>());
        system_mf_storage->reinit(mapping, dof_handler, one_dof_zero_and_hanging_node_constraints, QGauss<1>(fe.degree + 1), additional_data);
        q_phi_matrix.initialize(system_mf_storage);
    }
    q_phi_matrix.initialize_dof_vector(q);
    q_phi_matrix.initialize_dof_vector(q_phi_rhs);
    q_phi_matrix.initialize_dof_vector(phi);

    {
        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_values | update_JxW_values);
        std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(new MatrixFree<dim, double>());
        system_mf_storage->reinit(mapping, dof_handler, hanging_node_constraints, QGauss<1>(fe.degree + 1), additional_data);
        vorticity_matrix.initialize(system_mf_storage);
    }
    vorticity_matrix.initialize_dof_vector(vorticity);
    vorticity_matrix.initialize_dof_vector(vorticity_rhs);

    const unsigned int nlevels = triangulation.n_global_levels();
    u_mg_matrices.resize(0, nlevels - 1);
    w_mg_matrices.resize(0, nlevels - 1);
    q_phi_mg_matrices.resize(0, nlevels - 1);
    vorticity_mg_matrices.resize(0, nlevels - 1);

    u_w_mg_constrained_dofs.clear();
    u_w_mg_constrained_dofs.initialize(dof_handler);
    const std::set<types::boundary_id> dirichlet_boundary_ids = {0};
    u_w_mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary_ids);

    q_phi_mg_constrained_dofs.clear();
    q_phi_mg_constrained_dofs.initialize(dof_handler);

    vorticity_mg_constrained_dofs.clear();
    vorticity_mg_constrained_dofs.initialize(dof_handler);

    for (unsigned int level = 0; level < nlevels; ++level)
    {
        const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        // level_constraints.add_lines(u_w_mg_constrained_dofs.get_refinement_edge_indices(level));
        level_constraints.add_lines(u_w_mg_constrained_dofs.get_boundary_indices(level));
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_values | update_JxW_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
        mg_mf_storage_level->reinit(mapping, dof_handler, level_constraints, QGauss<1>(fe.degree + 1), additional_data);
        u_mg_matrices[level].initialize(mg_mf_storage_level, u_w_mg_constrained_dofs, level);

        // u_mg_matrices[level].compute_diagonal();
    }
    for (unsigned int level = 0; level < nlevels; ++level)
    {
        const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        // level_constraints.add_lines(u_w_mg_constrained_dofs.get_refinement_edge_indices(level));
        level_constraints.add_lines(u_w_mg_constrained_dofs.get_boundary_indices(level));
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
        mg_mf_storage_level->reinit(mapping, dof_handler, level_constraints, QGauss<1>(fe.degree + 1), additional_data);
        w_mg_matrices[level].initialize(mg_mf_storage_level, u_w_mg_constrained_dofs, level);

        w_mg_matrices[level].evaluate_coefficient();
        // w_mg_matrices[level].compute_diagonal();
    }
    for (unsigned int level = 0; level < nlevels; ++level)
    {
        const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        // level_constraints.add_lines(q_phi_mg_constrained_dofs.get_refinement_edge_indices(level));
        if (level_constraints.can_store_line(0))
            // if (relevant_dofs.is_element(0))
            level_constraints.add_line(0);
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
        mg_mf_storage_level->reinit(mapping, dof_handler, level_constraints, QGauss<1>(fe.degree + 1), additional_data);
        q_phi_mg_matrices[level].initialize(mg_mf_storage_level, q_phi_mg_constrained_dofs, level);

        // q_phi_mg_matrices[level].compute_diagonal();
    }
    for (unsigned int level = 0; level < nlevels; ++level)
    {
        const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_values | update_JxW_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
        mg_mf_storage_level->reinit(mapping, dof_handler, level_constraints, QGauss<1>(fe.degree + 1), additional_data);
        vorticity_mg_matrices[level].initialize(mg_mf_storage_level, vorticity_mg_constrained_dofs, level);

        // vorticity_mg_matrices[level].compute_diagonal();
    }
}

template <int dim>
void GePUP<dim>::initialize_u()
{
    TimerOutput::Scope t(computing_timer, "initialize u");

    velocity_value.set_time(0);
    for (unsigned int d = 0; d < dim; ++d)
    {
        u[d].zero_out_ghost_values();

        velocity_value.set_component(d);
        VectorTools::interpolate(mapping, dof_handler, velocity_value, u[d]);
        hanging_node_constraints.distribute(u[d]);

        u[d].update_ghost_values();
    }
}

template <int dim>
void GePUP<dim>::compute_old_w_integral()
{
    TimerOutput::Scope t(computing_timer, "compute old_w_integral");

    for (unsigned int d = 0; d < dim; ++d)
        w[d].update_ghost_values();

    for (unsigned int d = 0; d < dim; ++d)
        old_w_integral[d] = 0;

    FEValues<dim> fe_values(mapping, fe, QGauss<dim>(fe.degree + 1), update_values | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = fe_values.get_quadrature().size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> old_w_value[dim];
    Vector<double> cell_rhs[dim];
    for (unsigned int d = 0; d < dim; ++d)
    {
        old_w_value[d].resize(n_q_points);
        cell_rhs[d].reinit(dofs_per_cell);
    }
    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell->get_dof_indices(local_dof_indices);
            fe_values.reinit(cell);
            for (unsigned int d = 0; d < dim; ++d)
                cell_rhs[d] = 0;

            for (unsigned int d = 0; d < dim; ++d)
                fe_values.get_function_values(w[d], old_w_value[d]);
            for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    double value_J_W = fe_values.shape_value(i, q) * fe_values.JxW(q);
                    for (unsigned int d = 0; d < dim; ++d)
                        cell_rhs[d](i) += old_w_value[d][q] * value_J_W;
                }
            for (unsigned int d = 0; d < dim; ++d)
                hanging_node_constraints.distribute_local_to_global(cell_rhs[d], local_dof_indices, old_w_integral[d]);
        }
    for (unsigned int d = 0; d < dim; ++d)
        old_w_integral[d].compress(VectorOperation::add);
}

template <int dim>
void GePUP<dim>::compute_q_and_get_minus_u_dot_grad_u_integral(double current_time, unsigned int step)
{
    TimerOutput::Scope t(computing_timer, "compute q and get minus_u_dot_grad_u_integral");

    for (unsigned int d = 0; d < dim; ++d)
        u[d].update_ghost_values();

    for (unsigned int d = 0; d < dim; ++d)
        ex_integral[d][step] = 0;
    q_phi_rhs = 0;

    velocity_boundary.set_time(current_time);
    velocity_boundary_t.set_time(current_time);
    body_force.set_time(current_time);

    FEValues<dim> fe_values(mapping, fe, QGauss<dim>(fe.degree + 2), update_values | update_quadrature_points | update_gradients | update_hessians | update_JxW_values);
    FEFaceValues<dim> fe_face_values(mapping, fe, QGauss<dim - 1>(fe.degree + 2), update_values | update_quadrature_points | update_gradients | update_hessians | update_normal_vectors | update_JxW_values);
    const unsigned int n_q_points = fe_values.get_quadrature().size();
    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> div_of_g(n_q_points);
    std::vector<double> u_value[dim];
    std::vector<Tensor<1, dim>> grad_u[dim];
    std::vector<Tensor<2, dim>> hessian_u[dim];
    std::vector<Tensor<1, dim>> u_dot_grad_u(n_q_points);
    std::vector<double> div_of_u_dot_grad_u(n_q_points);

    std::vector<double> boundary_g_value[dim];
    std::vector<double> boundary_u_value[dim];
    std::vector<Tensor<1, dim>> boundary_grad_u[dim];
    std::vector<Tensor<2, dim>> boundary_hessian_u[dim];
    std::vector<double> boundary_u_t[dim];
    std::vector<Tensor<1, dim>> boundary_u_dot_grad_u(n_face_q_points);
    std::vector<Tensor<1, dim>> boundary_laplace_u_minus_grad_of_div_u(n_face_q_points);

    Vector<double> cell_rhs(dofs_per_cell);
    Vector<double> cell_u_dot_grad_u[dim];
    for (unsigned int d = 0; d < dim; ++d)
    {
        cell_u_dot_grad_u[d].reinit(dofs_per_cell);

        u_value[d].resize(n_q_points);
        grad_u[d].resize(n_q_points);
        hessian_u[d].resize(n_q_points);

        boundary_g_value[d].resize(n_face_q_points);
        boundary_u_value[d].resize(n_face_q_points);
        boundary_grad_u[d].resize(n_face_q_points);
        boundary_hessian_u[d].resize(n_face_q_points);
        boundary_u_t[d].resize(n_face_q_points);
    }
    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            cell_rhs = 0;

            body_force.divergence_list(fe_values.get_quadrature_points(), div_of_g);
            for (unsigned int d = 0; d < dim; ++d)
            {
                cell_u_dot_grad_u[d] = 0;
                fe_values.get_function_values(u[d], u_value[d]);
                fe_values.get_function_gradients(u[d], grad_u[d]);
                fe_values.get_function_hessians(u[d], hessian_u[d]);
            }
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                u_dot_grad_u[q] = 0;
                div_of_u_dot_grad_u[q] = 0;
                for (unsigned int d = 0; d < dim; ++d)
                    for (unsigned int d1 = 0; d1 < dim; ++d1)
                    {
                        u_dot_grad_u[q][d] += u_value[d1][q] * grad_u[d][q][d1];
                        div_of_u_dot_grad_u[q] += grad_u[d][q][d1] * grad_u[d1][q][d] + u_value[d][q] * hessian_u[d1][q][d][d1];
                    }
            }
            for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    double value_J_W = fe_values.shape_value(i, q) * fe_values.JxW(q);
                    cell_rhs(i) += (div_of_u_dot_grad_u[q] - div_of_g[q]) * value_J_W;
                    for (unsigned int d = 0; d < dim; ++d)
                        cell_u_dot_grad_u[d](i) -= u_dot_grad_u[q][d] * value_J_W;
                }

            for (const auto &face : cell->face_iterators())
                if (face->at_boundary())
                {
                    fe_face_values.reinit(cell, face);
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        body_force.set_component(d);
                        velocity_boundary_t.set_component(d);
                        body_force.value_list(fe_face_values.get_quadrature_points(), boundary_g_value[d]);
                        velocity_boundary_t.value_list(fe_face_values.get_quadrature_points(), boundary_u_t[d]);
                        fe_face_values.get_function_values(u[d], boundary_u_value[d]);
                        fe_face_values.get_function_gradients(u[d], boundary_grad_u[d]);
                        fe_face_values.get_function_hessians(u[d], boundary_hessian_u[d]);
                    }
                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                        boundary_u_dot_grad_u[q] = 0;
                        boundary_laplace_u_minus_grad_of_div_u[q] = 0;
                        for (unsigned int d = 0; d < dim; ++d)
                            for (unsigned int d1 = 0; d1 < dim; ++d1)
                            {
                                boundary_u_dot_grad_u[q][d] += boundary_u_value[d1][q] * boundary_grad_u[d][q][d1];
                                if (d1 != d)
                                    boundary_laplace_u_minus_grad_of_div_u[q][d] += boundary_hessian_u[d][q][d1][d1] - boundary_hessian_u[d1][q][d][d1];
                            }
                        boundary_u_dot_grad_u[q] = vis * boundary_laplace_u_minus_grad_of_div_u[q] - boundary_u_dot_grad_u[q];
                        for (unsigned int d = 0; d < dim; ++d)
                            boundary_u_dot_grad_u[q][d] += boundary_g_value[d][q] - boundary_u_t[d][q];
                    }
                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                        double rhs_value = boundary_u_dot_grad_u[q] * fe_face_values.normal_vector(q);
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            cell_rhs(i) += rhs_value * fe_face_values.shape_value(i, q) * fe_face_values.JxW(q);
                    }
                }
            hanging_node_constraints.distribute_local_to_global(cell_rhs, local_dof_indices, q_phi_rhs);
            for (unsigned int d = 0; d < dim; ++d)
                hanging_node_constraints.distribute_local_to_global(cell_u_dot_grad_u[d], local_dof_indices, ex_integral[d][step]);
        }
    for (unsigned int d = 0; d < dim; ++d)
        ex_integral[d][step].compress(VectorOperation::add);
    q_phi_rhs.compress(VectorOperation::add);

    q_phi_rhs.add(-q_phi_rhs.mean_value());

    {
        TimerOutput::Scope t(computing_timer, "slove q");

        MGTransferMatrixFree<dim, float> mg_transfer(q_phi_mg_constrained_dofs);
        mg_transfer.build(dof_handler);

        using SmootherType = PreconditionChebyshev<LaplaceLevelMatrixType, LinearAlgebra::distributed::Vector<float>>;
        mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        smoother_data.resize(0, triangulation.n_global_levels() - 1);

        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        {
            if (level > 0)
            {
                smoother_data[level].smoothing_range = 15.;
                smoother_data[level].degree = 5;
                smoother_data[level].eig_cg_n_iterations = 10;
            }
            else
            {
                smoother_data[0].smoothing_range = 1e-3;
                smoother_data[0].degree = numbers::invalid_unsigned_int;
                smoother_data[0].eig_cg_n_iterations = q_phi_mg_matrices[0].m();
            }
            q_phi_mg_matrices[level].compute_diagonal();
            smoother_data[level].preconditioner = q_phi_mg_matrices[level].get_matrix_diagonal_inverse();
        }
        mg_smoother.initialize(q_phi_mg_matrices, smoother_data);

        MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
        mg_coarse.initialize(mg_smoother);

        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(q_phi_mg_matrices);

        MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LaplaceLevelMatrixType>> mg_interface_matrices;
        mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            mg_interface_matrices[level].initialize(q_phi_mg_matrices[level]);
        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(mg_interface_matrices);

        Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
        mg.set_edge_matrices(mg_interface, mg_interface);

        PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>, MGTransferMatrixFree<dim, float>>
            preconditioner(dof_handler, mg, mg_transfer);

        one_dof_zero_and_hanging_node_constraints.set_zero(q);
        one_dof_zero_and_hanging_node_constraints.set_zero(q_phi_rhs);
        SolverControl solver_control(1000, 1e-12);
        // SolverMinRes<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        {
            TimerOutput::Scope t(computing_timer, "pure_solve_q");
            cg.solve(q_phi_matrix, q, q_phi_rhs, preconditioner);
        }
        one_dof_zero_and_hanging_node_constraints.distribute(q);
        q.add(-q.mean_value());
        q.update_ghost_values();

        pcout << "Number of iterations for solving q: " << solver_control.last_step() << std::endl;
    }
}

template <int dim>
void GePUP<dim>::compute_ex_integral_and_laplacian_w_integral(double current_time, unsigned int step)
{
    TimerOutput::Scope t(computing_timer, "compute ex_integral and laplacian_w_integral");

    for (unsigned int d = 0; d < dim; ++d)
        laplacian_w_integral[d][step] = 0;
    body_force.set_time(current_time);

    FEValues<dim> fe_values(mapping, fe, QGauss<dim>(fe.degree + 1), update_values | update_gradients | update_hessians | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = fe_values.get_quadrature().size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> g_value[dim];
    std::vector<Tensor<1, dim>> grad_q(n_q_points);
    std::vector<double> laplacian_w[dim];

    Vector<double> cell_ex[dim];
    Vector<double> cell_laplacian_w[dim];
    for (unsigned int d = 0; d < dim; ++d)
    {
        g_value[d].resize(n_q_points);
        laplacian_w[d].resize(n_q_points);

        cell_ex[d].reinit(dofs_per_cell);
        cell_laplacian_w[d].reinit(dofs_per_cell);
    }
    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell->get_dof_indices(local_dof_indices);
            fe_values.reinit(cell);
            for (unsigned int d = 0; d < dim; ++d)
            {
                cell_ex[d] = 0;
                cell_laplacian_w[d] = 0;
            }
            for (unsigned int d = 0; d < dim; ++d)
            {
                body_force.set_component(d);
                body_force.value_list(fe_values.get_quadrature_points(), g_value[d]);
                fe_values.get_function_laplacians(w[d], laplacian_w[d]);
            }
            fe_values.get_function_gradients(q, grad_q);
            for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    double value_J_W = fe_values.shape_value(i, q) * fe_values.JxW(q);
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        cell_ex[d](i) += (g_value[d][q] - grad_q[q][d]) * value_J_W;
                        cell_laplacian_w[d](i) += laplacian_w[d][q] * value_J_W;
                    }
                }
            for (unsigned int d = 0; d < dim; ++d)
            {
                hanging_node_constraints.distribute_local_to_global(cell_ex[d], local_dof_indices, ex_integral[d][step]);
                // hanging_node_constraints.distribute_local_to_global(cell_laplacian_w[d], local_dof_indices, laplacian_w_integral[d][step]);
            }
        }
    for (unsigned int d = 0; d < dim; ++d)
    {
        ex_integral[d][step].compress(VectorOperation::add);
        // laplacian_w_integral[d][step].compress(VectorOperation::add);
    }

    LaplaceMatrixType inhomogeneous_operator;
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values);
    std::shared_ptr<MatrixFree<dim, double>> matrix_free(new MatrixFree<dim, double>());
    matrix_free->reinit(mapping, dof_handler, hanging_node_constraints, QGauss<1>(fe.degree + 1), additional_data);
    inhomogeneous_operator.initialize(matrix_free);
    inhomogeneous_operator.initialize_dof_vector(tmp_vec);
    inhomogeneous_operator.initialize_dof_vector(tmp_rhs_vec);
    for (unsigned int d = 0; d < dim; ++d)
    {
        tmp_vec = w[d];
        inhomogeneous_operator.vmult(tmp_rhs_vec, tmp_vec);
        tmp_rhs_vec *= -1.;
        laplacian_w_integral[d][step] = tmp_rhs_vec;
    }
}

template <int dim>
void GePUP<dim>::compute_w(double current_time, unsigned int step)
{
    TimerOutput::Scope t(computing_timer, "compute w");

    velocity_boundary.set_time(current_time);

    HelmholtzMatrixType inhomogeneous_operator;
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values);
    std::shared_ptr<MatrixFree<dim, double>> matrix_free(new MatrixFree<dim, double>());
    matrix_free->reinit(mapping, dof_handler, hanging_node_constraints, QGauss<1>(fe.degree + 1), additional_data);
    inhomogeneous_operator.initialize(matrix_free);
    inhomogeneous_operator.initialize_dof_vector(tmp_vec);
    inhomogeneous_operator.initialize_dof_vector(tmp_rhs_vec);

    AffineConstraints<double> inhomogeneous_boundary_constraints[dim];
    for (unsigned int d = 0; d < dim; ++d)
    {
        velocity_boundary.set_component(d);
        inhomogeneous_boundary_constraints[d].clear();
        inhomogeneous_boundary_constraints[d].reinit(locally_relevant_dofs);
        VectorTools::interpolate_boundary_values(mapping, dof_handler, 0, velocity_boundary, inhomogeneous_boundary_constraints[d]);
        inhomogeneous_boundary_constraints[d].close();
        tmp_vec = 0;
        inhomogeneous_boundary_constraints[d].distribute(tmp_vec);
        inhomogeneous_operator.vmult(tmp_rhs_vec, tmp_vec);
        tmp_rhs_vec *= -1;
        w_rhs[d] = tmp_rhs_vec;
    }

    for (unsigned int d = 0; d < dim; ++d)
    {
        w_rhs[d] += old_w_integral[d];
        for (unsigned int s = 0; s < step; ++s)
        {
            w_rhs[d].add(time_step * a_ex[step][s], ex_integral[d][s]);
            w_rhs[d].add(time_step * vis * a_im[step][s], laplacian_w_integral[d][s]);
        }
    }

    for (unsigned int d = 0; d < dim; ++d)
    {

        MGTransferMatrixFree<dim, float> mg_transfer(u_w_mg_constrained_dofs);
        mg_transfer.build(dof_handler);

        using SmootherType = PreconditionChebyshev<HelmholtzLevelMatrixType, LinearAlgebra::distributed::Vector<float>>;
        mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        smoother_data.resize(0, triangulation.n_global_levels() - 1);

        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        {
            if (level > 0)
            {
                smoother_data[level].smoothing_range = 15.;
                smoother_data[level].degree = 5;
                smoother_data[level].eig_cg_n_iterations = 10;
            }
            else
            {
                smoother_data[0].smoothing_range = 1e-3;
                smoother_data[0].degree = numbers::invalid_unsigned_int;
                smoother_data[0].eig_cg_n_iterations = w_mg_matrices[0].m();
            }
            w_mg_matrices[level].compute_diagonal();
            smoother_data[level].preconditioner = w_mg_matrices[level].get_matrix_diagonal_inverse();
        }
        mg_smoother.initialize(w_mg_matrices, smoother_data);

        MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
        mg_coarse.initialize(mg_smoother);

        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(w_mg_matrices);

        MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<HelmholtzLevelMatrixType>> mg_interface_matrices;
        mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            mg_interface_matrices[level].initialize(w_mg_matrices[level]);
        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(mg_interface_matrices);

        Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
        mg.set_edge_matrices(mg_interface, mg_interface);

        PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>, MGTransferMatrixFree<dim, float>>
            preconditioner(dof_handler, mg, mg_transfer);

        zero_boundary_and_hanging_node_constraints.set_zero(w[d]);
        zero_boundary_and_hanging_node_constraints.set_zero(w_rhs[d]);
        SolverControl solver_control(1000, 1e-12);
        // SolverMinRes<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        {
            TimerOutput::Scope t(computing_timer, "pure_solve_w");
            cg.solve(w_matrix, w[d], w_rhs[d], preconditioner);
        }
        inhomogeneous_boundary_constraints[d].distribute(w[d]);
        hanging_node_constraints.distribute(w[d]);
        w[d].update_ghost_values();

        pcout << "Number of iterations for solving w: " << solver_control.last_step() << std::endl;
    }
}

template <int dim>
void GePUP<dim>::compute_phi(double current_time)
{
    TimerOutput::Scope t(computing_timer, "compute phi");

    velocity_boundary.set_time(current_time);
    q_phi_rhs = 0;

    FEValues<dim> fe_values(mapping, fe, QGauss<dim>(fe.degree + 1), update_values | update_gradients | update_JxW_values);
    FEFaceValues<dim> fe_face_values(mapping, fe, QGauss<dim - 1>(fe.degree + 1), update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = fe_values.get_quadrature().size();
    const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> w_value[dim];
    std::vector<Tensor<1, dim>> grad_w[dim];
    std::vector<double> div_w(n_q_points);
    std::vector<double> boundary_w_value[dim];
    std::vector<double> boundary_u_value[dim];
    for (unsigned int d = 0; d < dim; ++d)
    {
        w_value[d].resize(n_q_points);
        grad_w[d].resize(n_q_points);
        boundary_w_value[d].resize(n_face_q_points);
        boundary_u_value[d].resize(n_face_q_points);
    }
    Vector<double> cell_rhs(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell->get_dof_indices(local_dof_indices);
            fe_values.reinit(cell);
            cell_rhs = 0;
            for (unsigned int d = 0; d < dim; ++d)
                fe_values.get_function_gradients(w[d], grad_w[d]);
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                div_w[q] = 0;
                for (unsigned int d = 0; d < dim; ++d)
                    div_w[q] += grad_w[d][q][d];
            }
            for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_rhs(i) -= div_w[q] * fe_values.shape_value(i, q) * fe_values.JxW(q);
            for (const auto &face : cell->face_iterators())
                if (face->at_boundary())
                {
                    fe_face_values.reinit(cell, face);
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        velocity_boundary.set_component(d);
                        velocity_boundary.value_list(fe_face_values.get_quadrature_points(), boundary_u_value[d]);
                        fe_face_values.get_function_values(w[d], boundary_w_value[d]);
                    }
                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                        double rhs_boundary_value = 0;
                        for (unsigned int d = 0; d < dim; ++d)
                            rhs_boundary_value += fe_face_values.normal_vector(q)[d] * (boundary_w_value[d][q] - boundary_u_value[d][q]);
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            cell_rhs(i) += rhs_boundary_value * fe_face_values.shape_value(i, q) * fe_face_values.JxW(q);
                    }
                }
            hanging_node_constraints.distribute_local_to_global(cell_rhs, local_dof_indices, q_phi_rhs);
        }
    q_phi_rhs.compress(VectorOperation::add);
    q_phi_rhs.add(-q_phi_rhs.mean_value());

    {
        TimerOutput::Scope t(computing_timer, "slove phi");

        MGTransferMatrixFree<dim, float> mg_transfer(q_phi_mg_constrained_dofs);
        mg_transfer.build(dof_handler);

        using SmootherType = PreconditionChebyshev<LaplaceLevelMatrixType, LinearAlgebra::distributed::Vector<float>>;
        mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        smoother_data.resize(0, triangulation.n_global_levels() - 1);

        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        {
            if (level > 0)
            {
                smoother_data[level].smoothing_range = 15.;
                smoother_data[level].degree = 5;
                smoother_data[level].eig_cg_n_iterations = 10;
            }
            else
            {
                smoother_data[0].smoothing_range = 1e-3;
                smoother_data[0].degree = numbers::invalid_unsigned_int;
                smoother_data[0].eig_cg_n_iterations = q_phi_mg_matrices[0].m();
            }
            q_phi_mg_matrices[level].compute_diagonal();
            smoother_data[level].preconditioner = q_phi_mg_matrices[level].get_matrix_diagonal_inverse();
        }
        mg_smoother.initialize(q_phi_mg_matrices, smoother_data);

        MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
        mg_coarse.initialize(mg_smoother);

        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(q_phi_mg_matrices);

        MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LaplaceLevelMatrixType>> mg_interface_matrices;
        mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            mg_interface_matrices[level].initialize(q_phi_mg_matrices[level]);
        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(mg_interface_matrices);

        Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
        mg.set_edge_matrices(mg_interface, mg_interface);

        PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>, MGTransferMatrixFree<dim, float>>
            preconditioner(dof_handler, mg, mg_transfer);

        one_dof_zero_and_hanging_node_constraints.set_zero(phi);
        one_dof_zero_and_hanging_node_constraints.set_zero(q_phi_rhs);
        SolverControl solver_control(1000, 1e-12);
        // SolverMinRes<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        {
            TimerOutput::Scope t(computing_timer, "pure_solve_phi");
            cg.solve(q_phi_matrix, phi, q_phi_rhs, preconditioner);
        }
        one_dof_zero_and_hanging_node_constraints.distribute(phi);

        phi.add(-phi.mean_value());
        phi.update_ghost_values();

        pcout << "Number of iterations for solving phi: " << solver_control.last_step() << std::endl;
    }
}

template <int dim>
void GePUP<dim>::compute_u(double current_time)
{
    TimerOutput::Scope t(computing_timer, "compute u");

    velocity_boundary.set_time(current_time);

    MassMatrixType inhomogeneous_operator;
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = (update_values | update_JxW_values);
    std::shared_ptr<MatrixFree<dim, double>> matrix_free(new MatrixFree<dim, double>());
    matrix_free->reinit(mapping, dof_handler, hanging_node_constraints, QGauss<1>(fe.degree + 1), additional_data);
    inhomogeneous_operator.initialize(matrix_free);
    inhomogeneous_operator.initialize_dof_vector(tmp_vec);
    inhomogeneous_operator.initialize_dof_vector(tmp_rhs_vec);

    AffineConstraints<double> inhomogeneous_boundary_constraints[dim];
    for (unsigned int d = 0; d < dim; ++d)
    {
        velocity_boundary.set_component(d);
        inhomogeneous_boundary_constraints[d].clear();
        inhomogeneous_boundary_constraints[d].reinit(locally_relevant_dofs);
        VectorTools::interpolate_boundary_values(mapping, dof_handler, 0, velocity_boundary, inhomogeneous_boundary_constraints[d]);
        inhomogeneous_boundary_constraints[d].close();
        tmp_vec = 0;
        inhomogeneous_boundary_constraints[d].distribute(tmp_vec);
        inhomogeneous_operator.vmult(tmp_rhs_vec, tmp_vec);
        tmp_rhs_vec *= -1;
        u_rhs[d] = tmp_rhs_vec;
    }

    FEValues<dim> fe_values(mapping, fe, QGauss<dim>(fe.degree + 1), update_values | update_gradients | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = fe_values.get_quadrature().size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> w_value[dim];
    std::vector<Tensor<1, dim>> grad_phi(n_q_points);
    Vector<double> cell_rhs[dim];
    for (unsigned int d = 0; d < dim; ++d)
    {
        cell_rhs[d].reinit(dofs_per_cell);
        w_value[d].resize(n_q_points);
    }
    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell->get_dof_indices(local_dof_indices);
            fe_values.reinit(cell);
            for (unsigned int d = 0; d < dim; ++d)
                cell_rhs[d] = 0;
            fe_values.get_function_gradients(phi, grad_phi);
            for (unsigned int d = 0; d < dim; ++d)
                fe_values.get_function_values(w[d], w_value[d]);
            for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    double value_J_W = fe_values.shape_value(i, q) * fe_values.JxW(q);
                    for (unsigned int d = 0; d < dim; ++d)
                        cell_rhs[d](i) += (w_value[d][q] - grad_phi[q][d]) * value_J_W;
                }
            for (unsigned int d = 0; d < dim; ++d)
                hanging_node_constraints.distribute_local_to_global(cell_rhs[d], local_dof_indices, u_rhs[d]);
        }
    for (unsigned int d = 0; d < dim; ++d)
        u_rhs[d].compress(VectorOperation::add);

    for (unsigned int d = 0; d < dim; ++d)
    {
        TimerOutput::Scope t(computing_timer, "slove u");

        MGTransferMatrixFree<dim, float> mg_transfer(u_w_mg_constrained_dofs);
        mg_transfer.build(dof_handler);

        using SmootherType = PreconditionChebyshev<MassLevelMatrixType, LinearAlgebra::distributed::Vector<float>>;
        mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        smoother_data.resize(0, triangulation.n_global_levels() - 1);

        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        {
            if (level > 0)
            {
                smoother_data[level].smoothing_range = 15.;
                smoother_data[level].degree = 5;
                smoother_data[level].eig_cg_n_iterations = 10;
            }
            else
            {
                smoother_data[0].smoothing_range = 1e-3;
                smoother_data[0].degree = numbers::invalid_unsigned_int;
                smoother_data[0].eig_cg_n_iterations = u_mg_matrices[0].m();
            }
            u_mg_matrices[level].compute_diagonal();
            smoother_data[level].preconditioner = u_mg_matrices[level].get_matrix_diagonal_inverse();
        }
        mg_smoother.initialize(u_mg_matrices, smoother_data);

        MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
        mg_coarse.initialize(mg_smoother);

        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(u_mg_matrices);

        MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<MassLevelMatrixType>> mg_interface_matrices;
        mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            mg_interface_matrices[level].initialize(u_mg_matrices[level]);
        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(mg_interface_matrices);

        Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
        mg.set_edge_matrices(mg_interface, mg_interface);

        PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>, MGTransferMatrixFree<dim, float>>
            preconditioner(dof_handler, mg, mg_transfer);

        zero_boundary_and_hanging_node_constraints.set_zero(u[d]);
        zero_boundary_and_hanging_node_constraints.set_zero(u_rhs[d]);
        SolverControl solver_control(1000, 1e-12);
        // SolverMinRes<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        {
            TimerOutput::Scope t(computing_timer, "pure_solve_u");
            cg.solve(u_matrix, u[d], u_rhs[d], preconditioner);
        }
        inhomogeneous_boundary_constraints[d].distribute(u[d]);
        hanging_node_constraints.distribute(u[d]);
        u[d].update_ghost_values();

        pcout << "Number of iterations for solving u: " << solver_control.last_step() << std::endl;
    }
}

template <int dim>
void GePUP<dim>::compute_w_star(double current_time)
{
    TimerOutput::Scope t(computing_timer, "compute w_star");

    body_force.set_time(current_time);
    velocity_boundary.set_time(current_time);

    MassMatrixType inhomogeneous_operator;
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = (update_values | update_JxW_values);
    std::shared_ptr<MatrixFree<dim, double>> matrix_free(new MatrixFree<dim, double>());
    matrix_free->reinit(mapping, dof_handler, hanging_node_constraints, QGauss<1>(fe.degree + 1), additional_data);
    inhomogeneous_operator.initialize(matrix_free);
    inhomogeneous_operator.initialize_dof_vector(tmp_vec);
    inhomogeneous_operator.initialize_dof_vector(tmp_rhs_vec);

    AffineConstraints<double> inhomogeneous_boundary_constraints[dim];
    for (unsigned int d = 0; d < dim; ++d)
    {
        velocity_boundary.set_component(d);
        inhomogeneous_boundary_constraints[d].clear();
        inhomogeneous_boundary_constraints[d].reinit(locally_relevant_dofs);
        VectorTools::interpolate_boundary_values(mapping, dof_handler, 0, velocity_boundary, inhomogeneous_boundary_constraints[d]);
        inhomogeneous_boundary_constraints[d].close();
        tmp_vec = 0;
        inhomogeneous_boundary_constraints[d].distribute(tmp_vec);
        inhomogeneous_operator.vmult(tmp_rhs_vec, tmp_vec);
        tmp_rhs_vec *= -1;
        u_rhs[d] = tmp_rhs_vec;
    }

    FEValues<dim> fe_values(mapping, fe, QGauss<dim>(fe.degree + 1), update_values | update_gradients | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = fe_values.get_quadrature().size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> w_value[dim];
    std::vector<double> g_value[dim];
    std::vector<Tensor<1, dim>> grad_q(n_q_points);
    Vector<double> cell_rhs[dim];
    Vector<double> cell_ex[dim];
    for (unsigned int d = 0; d < dim; ++d)
    {
        w_value[d].resize(n_q_points);
        g_value[d].resize(n_q_points);
        cell_rhs[d].reinit(dofs_per_cell);
        cell_ex[d].reinit(dofs_per_cell);
    }
    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell->get_dof_indices(local_dof_indices);
            fe_values.reinit(cell);
            for (unsigned int d = 0; d < dim; ++d)
            {
                cell_rhs[d] = 0;
                cell_ex[d] = 0;
            }
            for (unsigned int d = 0; d < dim; ++d)
            {
                body_force.set_component(d);
                body_force.value_list(fe_values.get_quadrature_points(), g_value[d]);

                fe_values.get_function_values(w[d], w_value[d]);
            }
            fe_values.get_function_gradients(q, grad_q);
            for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    double value_J_W = fe_values.shape_value(i, q) * fe_values.JxW(q);
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        cell_rhs[d](i) += w_value[d][q] * value_J_W;
                        cell_ex[d](i) += (g_value[d][q] - grad_q[q][d]) * value_J_W;
                    }
                }
            for (unsigned int d = 0; d < dim; ++d)
            {
                hanging_node_constraints.distribute_local_to_global(cell_rhs[d], local_dof_indices, u_rhs[d]);
                hanging_node_constraints.distribute_local_to_global(cell_ex[d], local_dof_indices, ex_integral[d][5]);
            }
        }
    for (unsigned int d = 0; d < dim; ++d)
    {
        u_rhs[d].compress(VectorOperation::add);
        ex_integral[d][5].compress(VectorOperation::add);
    }
    for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int s = 0; s < 6; ++s)
            u_rhs[d].add(time_step * (b[s] - a_ex[5][s]), ex_integral[d][s]);

    for (unsigned int d = 0; d < dim; ++d)
    {
        TimerOutput::Scope t(computing_timer, "slove w star");

        MGTransferMatrixFree<dim, float> mg_transfer(u_w_mg_constrained_dofs);
        mg_transfer.build(dof_handler);

        using SmootherType = PreconditionChebyshev<MassLevelMatrixType, LinearAlgebra::distributed::Vector<float>>;
        mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        smoother_data.resize(0, triangulation.n_global_levels() - 1);

        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        {
            if (level > 0)
            {
                smoother_data[level].smoothing_range = 15.;
                smoother_data[level].degree = 5;
                smoother_data[level].eig_cg_n_iterations = 10;
            }
            else
            {
                smoother_data[0].smoothing_range = 1e-3;
                smoother_data[0].degree = numbers::invalid_unsigned_int;
                smoother_data[0].eig_cg_n_iterations = u_mg_matrices[0].m();
            }
            u_mg_matrices[level].compute_diagonal();
            smoother_data[level].preconditioner = u_mg_matrices[level].get_matrix_diagonal_inverse();
        }
        mg_smoother.initialize(u_mg_matrices, smoother_data);

        MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
        mg_coarse.initialize(mg_smoother);

        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(u_mg_matrices);

        MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<MassLevelMatrixType>> mg_interface_matrices;
        mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            mg_interface_matrices[level].initialize(u_mg_matrices[level]);
        mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(mg_interface_matrices);

        Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
        mg.set_edge_matrices(mg_interface, mg_interface);

        PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>, MGTransferMatrixFree<dim, float>>
            preconditioner(dof_handler, mg, mg_transfer);

        zero_boundary_and_hanging_node_constraints.set_zero(u[d]);
        zero_boundary_and_hanging_node_constraints.set_zero(u_rhs[d]);
        SolverControl solver_control(1000, 1e-12);
        // SolverMinRes<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
        {
            TimerOutput::Scope t(computing_timer, "pure_solve_w_star");
            cg.solve(u_matrix, u[d], u_rhs[d], preconditioner);
        }
        inhomogeneous_boundary_constraints[d].distribute(u[d]);
        hanging_node_constraints.distribute(u[d]);
        w[d] = u[d];
        w[d].update_ghost_values();

        pcout << "Number of iterations for solving w_star: " << solver_control.last_step() << std::endl;
    }
}

template <int dim>
void GePUP<dim>::compute_vorticity()
{
    TimerOutput::Scope t(computing_timer, "compute vorticity");

    vorticity_rhs = 0;

    FEValues<dim> fe_values(mapping, fe, QGauss<dim>(fe.degree + 1), update_values | update_gradients | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = fe_values.get_quadrature().size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> grad_u0(n_q_points);
    std::vector<Tensor<1, dim>> grad_u1(n_q_points);
    Vector<double> cell_rhs(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell->get_dof_indices(local_dof_indices);
            fe_values.reinit(cell);
            cell_rhs = 0;

            fe_values.get_function_gradients(u[0], grad_u0);
            fe_values.get_function_gradients(u[1], grad_u1);

            for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_rhs(i) += fe_values.shape_value(i, q) * (grad_u0[q][1] - grad_u1[q][0]) * fe_values.JxW(q);
            hanging_node_constraints.distribute_local_to_global(cell_rhs, local_dof_indices, vorticity_rhs);
        }
    vorticity_rhs.compress(VectorOperation::add);

    MGTransferMatrixFree<dim, float> mg_transfer(vorticity_mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    using SmootherType = PreconditionChebyshev<MassLevelMatrixType, LinearAlgebra::distributed::Vector<float>>;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);

    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        if (level > 0)
        {
            smoother_data[level].smoothing_range = 15.;
            smoother_data[level].degree = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = vorticity_mg_matrices[0].m();
        }
        vorticity_mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner = vorticity_mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(vorticity_mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(vorticity_mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<MassLevelMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_interface_matrices[level].initialize(vorticity_mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>, MGTransferMatrixFree<dim, float>>
        preconditioner(dof_handler, mg, mg_transfer);

    hanging_node_constraints.set_zero(vorticity);
    hanging_node_constraints.set_zero(vorticity_rhs);
    SolverControl solver_control(1000, 1e-12);
    // SolverMinRes<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
    cg.solve(vorticity_matrix, vorticity, vorticity_rhs, preconditioner);

    hanging_node_constraints.distribute(vorticity);
    vorticity.update_ghost_values();

    pcout << "Number of iterations for solving vorticity: " << solver_control.last_step() << std::endl;
}

template <int dim>
void GePUP<dim>::compute_vorticity_3d()
{
    TimerOutput::Scope t(computing_timer, "compute vorticity");

    vorticity_rhs = 0;

    FEValues<dim> fe_values(mapping, fe, QGauss<dim>(fe.degree + 1), update_values | update_gradients | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = fe_values.get_quadrature().size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> grad_u0(n_q_points);
    std::vector<Tensor<1, dim>> grad_u1(n_q_points);
    std::vector<Tensor<1, dim>> grad_u2(n_q_points);
    Vector<double> cell_rhs(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell->get_dof_indices(local_dof_indices);
            fe_values.reinit(cell);
            cell_rhs = 0;

            fe_values.get_function_gradients(u[0], grad_u0);
            fe_values.get_function_gradients(u[1], grad_u1);
            fe_values.get_function_gradients(u[2], grad_u2);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                double w0 = grad_u2[q][1] - grad_u1[q][2];
                double w1 = grad_u0[q][2] - grad_u2[q][0];
                double w2 = grad_u1[q][0] - grad_u0[q][1];
                double W = sqrt(w0 * w0 + w1 * w1 + w2 * w2);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_rhs(i) += fe_values.shape_value(i, q) * W * fe_values.JxW(q);
            }
            hanging_node_constraints.distribute_local_to_global(cell_rhs, local_dof_indices, vorticity_rhs);
        }
    vorticity_rhs.compress(VectorOperation::add);

    MGTransferMatrixFree<dim, float> mg_transfer(vorticity_mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    using SmootherType = PreconditionChebyshev<MassLevelMatrixType, LinearAlgebra::distributed::Vector<float>>;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);

    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        if (level > 0)
        {
            smoother_data[level].smoothing_range = 15.;
            smoother_data[level].degree = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = vorticity_mg_matrices[0].m();
        }
        vorticity_mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner = vorticity_mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(vorticity_mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(vorticity_mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<MassLevelMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_interface_matrices[level].initialize(vorticity_mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>, MGTransferMatrixFree<dim, float>>
        preconditioner(dof_handler, mg, mg_transfer);

    hanging_node_constraints.set_zero(vorticity);
    hanging_node_constraints.set_zero(vorticity_rhs);
    SolverControl solver_control(1000, 1e-12);
    // SolverMinRes<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
    cg.solve(vorticity_matrix, vorticity, vorticity_rhs, preconditioner);

    hanging_node_constraints.distribute(vorticity);
    vorticity.update_ghost_values();

    pcout << "Number of iterations for solving vorticity: " << solver_control.last_step() << std::endl;
}

template <int dim>
void GePUP<dim>::run_once()
{
    for (unsigned int d = 0; d < dim; ++d)
        w[d] = u[d];
    compute_old_w_integral();
    for (unsigned int s = 0; s < 5; ++s)
    {
        compute_q_and_get_minus_u_dot_grad_u_integral(time + c[s] * time_step, s);
        compute_ex_integral_and_laplacian_w_integral(time + c[s] * time_step, s);
        compute_w(time + c[s + 1] * time_step, s + 1);
        compute_phi(time + c[s + 1] * time_step);
        compute_u(time + c[s + 1] * time_step);
    }
    compute_q_and_get_minus_u_dot_grad_u_integral(time + c[5] * time_step, 5);
    compute_w_star(time + c[5] * time_step);
    compute_phi(time + c[5] * time_step);
    compute_u(time + c[5] * time_step);

    if (dim == 2)
        compute_vorticity();
    else
        compute_vorticity_3d();
}

template <int dim>
double GePUP<dim>::get_cfl_number()
{
    TimerOutput::Scope t(computing_timer, "get cfl_number");

    for (unsigned int d = 0; d < dim; ++d)
        u[d].update_ghost_values();

    const QIterated<dim> quadrature_formula(QTrapezoid<1>(), fe.degree);
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> fe_values(fe, quadrature_formula, update_values);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);
    double max_local_cfl = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            for (unsigned int d = 0; d < dim; ++d)
            {
                std::vector<double> one_dim_values(n_q_points);
                fe_values.get_function_values(u[d], one_dim_values);
                for (unsigned int q = 0; q < n_q_points; ++q)
                    velocity_values[q][d] = one_dim_values[q];
            }
            double max_local_velocity = 1e-10;
            for (unsigned int q = 0; q < n_q_points; ++q)
                max_local_velocity = std::max(max_local_velocity, velocity_values[q].norm());
            max_local_cfl = std::max(max_local_cfl, max_local_velocity / cell->diameter());
        }

    return Utilities::MPI::max(max_local_cfl, MPI_COMM_WORLD);
}

template <int dim>
void GePUP<dim>::refine_mesh(const unsigned int min_grid_level, const unsigned int max_grid_level)
{
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>> trans(dof_handler);

    {
        TimerOutput::Scope t(computing_timer, "refine_mesh, part 1");

        /*
        //设置u(梯度)为误差指标
        std::vector<LinearAlgebra::distributed::Vector<double>> temp_solution(dim);
        std::vector<const LinearAlgebra::distributed::Vector<double> *> temp_u(dim);
        for (unsigned int d = 0; d < dim; ++d)
        {
            temp_solution[d].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
            temp_solution[d] = u[d];
            temp_u[d] = &temp_solution[d];
        }

        std::vector<Vector<float>> estimated_error(dim);
        std::vector<Vector<float> *> estimated_error_per_cell(dim);
        for (unsigned int d = 0; d < dim; ++d)
            estimated_error_per_cell[d] = &estimated_error[d];
        KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(fe.degree + 1), std::map<types::boundary_id, const Function<dim> *>(), temp_u, estimated_error_per_cell, ComponentMask(), nullptr, 0, triangulation.locally_owned_subdomain());

        for (unsigned int d = 0; d < dim; ++d)
            estimated_error[d].scale(estimated_error[d]);
        for (unsigned int d = 1; d < dim; ++d)
            estimated_error[0] += estimated_error[d];
        for (unsigned int i = 0; i < estimated_error[0].size(); ++i)
            estimated_error[0](i) = sqrt(estimated_error[0](i));
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, estimated_error[0], 0.8, 0.1);
        */

        /*
        //设置q(梯度)为误差指标
        LinearAlgebra::distributed::Vector<double> temp_q;
        temp_q.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        temp_q = q;
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(fe.degree + 1), std::map<types::boundary_id, const Function<dim> *>(), temp_q, estimated_error_per_cell, ComponentMask(), nullptr, 0, triangulation.locally_owned_subdomain());
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, estimated_error_per_cell, 0.8, 0.1);
        */

        // 设置涡量为误差指标

        Vector<double> estimated_error_per_cell(triangulation.n_active_cells());

        FEValues<dim> fe_values(mapping, fe, QGauss<dim>(fe.degree + 1), update_values);
        const unsigned int n_q_points = fe_values.get_quadrature().size();
        std::vector<double> vor_indicator(n_q_points);
        for (const auto &cell : dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                fe_values.reinit(cell);
                fe_values.get_function_values(vorticity, vor_indicator);
                double sum_vor = 0;
                for (unsigned int q = 0; q < n_q_points; ++q)
                    sum_vor += fabs(vor_indicator[q]);
                // estimated_error_per_cell(cell->global_active_cell_index()) = sum_vor;
                estimated_error_per_cell(cell->active_cell_index()) = sum_vor;
            }
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, estimated_error_per_cell, 0.8, 0.1);

        if (triangulation.n_levels() > max_grid_level)
            for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(max_grid_level); cell != triangulation.end(); ++cell)
                cell->clear_refine_flag();
        for (const auto &cell : triangulation.active_cell_iterators_on_level(min_grid_level))
            cell->clear_coarsen_flag();

        std::vector<const LinearAlgebra::distributed::Vector<double> *> x_velocity(dim);
        for (unsigned int d = 0; d < dim; ++d)
            x_velocity[d] = &u[d];

        triangulation.prepare_coarsening_and_refinement();
        trans.prepare_for_coarsening_and_refinement(x_velocity);
        triangulation.execute_coarsening_and_refinement();
    }

    setup_dofs();

    {
        TimerOutput::Scope t(computing_timer, "refine_mesh, part 2");

        std::vector<LinearAlgebra::distributed::Vector<double>> distributed_temp(dim);
        for (unsigned int d = 0; d < dim; ++d)
            distributed_temp[d].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        std::vector<LinearAlgebra::distributed::Vector<double> *> tmp(dim);
        for (unsigned int d = 0; d < dim; ++d)
            tmp[d] = &distributed_temp[d];

        // 加不加应该都行
        for (unsigned int d = 0; d < dim; ++d)
            distributed_temp[d].zero_out_ghost_values();

        trans.interpolate(tmp);

        // 加不加应该都行
        for (unsigned int d = 0; d < dim; ++d)
            distributed_temp[d].update_ghost_values();

        for (unsigned int d = 0; d < dim; ++d)
        {
            u[d] = distributed_temp[d];
            hanging_node_constraints.distribute(u[d]);
        }
    }
}

template <int dim>
void GePUP<dim>::print_error()
{
    TimerOutput::Scope t(computing_timer, "print_error");

    velocity_value.set_component(0);
    velocity_value.set_time(time);
    Vector<float> norm_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler, u[0], velocity_value, norm_per_cell, QGauss<dim>(fe.degree + 1), VectorTools::L2_norm);
    double norm = VectorTools::compute_global_error(triangulation, norm_per_cell, VectorTools::L2_norm);

    velocity_value.set_component(1);
    Vector<float> norm_per_cell2(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler, u[1], velocity_value, norm_per_cell2, QGauss<dim>(fe.degree + 1), VectorTools::L2_norm);
    double norm2 = VectorTools::compute_global_error(triangulation, norm_per_cell2, VectorTools::L2_norm);

    double error = norm * norm + norm2 * norm2;
    error = sqrt(error);

    pcout << "error: " << error << std::endl;
}

template <int dim>
void GePUP<dim>::output_image(std::string file_name, unsigned int output_count, double current_time)
{
    TimerOutput::Scope t(computing_timer, "output_image");

    DataOut<dim> data_out;
    DataOutBase::VtkFlags flags;
    // flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);
    if (dim == 2)
    {
        data_out.add_data_vector(u[0], "Vx");
        data_out.add_data_vector(u[1], "Vy");
        data_out.add_data_vector(vorticity, "vorticity");
    }
    else
    {
        data_out.add_data_vector(u[0], "Vx");
        data_out.add_data_vector(u[1], "Vy");
        data_out.add_data_vector(u[2], "Vz");
        data_out.add_data_vector(vorticity, "vorticity");
    }
    data_out.build_patches(mapping, fe.degree, DataOut<dim>::curved_inner_cells);
    // data_out.build_patches();
    // data_out.write_vtu_with_pvtu_record(file_name, "image", output_count, mpi_communicator, 4, 1);
    std::string filename = file_name + "realtime_" + std::to_string(current_time);
    // std::ofstream output(filename);
    // data_out.write_vtu(output, mpi_communicator, 4, 1);
    data_out.write_vtu_with_pvtu_record(filename, "_timestep", output_count, mpi_communicator, 4, 1);
}

template <int dim>
void GePUP<dim>::run()
{
    // double max_cfl_number = 0;
    std::string parentDir = "2D-NS/"; // 图片文件名称
    std::string outputDir = parentDir + "parameter_" + std::to_string(parameter) + "/"; 

    std::filesystem::path parentPath(parentDir);
    std::filesystem::path outputPath(outputDir);
    try {
            // Check if the parent directory exists, if not, create it
            if (!std::filesystem::exists(parentPath)) {
                if (std::filesystem::create_directories(parentPath)) {
                    std::cout << "Parent directory " << parentDir << " created successfully." << std::endl;
                } else {
                    std::cerr << "Failed to create parent directory " << parentDir << "." << std::endl;
                }
            } else {
                std::cout << "Parent directory " << parentDir << " already exists." << std::endl;
            }

            // Check if the output directory exists, if not, create it
            if (!std::filesystem::exists(outputPath)) {
                if (std::filesystem::create_directories(outputPath)) {
                    std::cout << "Output directory " << outputDir << " created successfully." << std::endl;
                } else {
                    std::cerr << "Failed to create output directory " << outputDir << "." << std::endl;
                }
            } else {
                std::cout << "Output directory " << outputDir << " already exists." << std::endl;
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        }

    make_grid();
    setup_dofs();
    initialize_u();
    pcout << "time step 0" << std::endl;
    print_error();
    output_image(outputDir, 0, 0);

    // double current_cfl_number = get_cfl_number();
    // max_cfl_number = std::max(max_cfl_number, current_cfl_number);
    // pcout << "current Cr number: " << current_cfl_number * time_step << std::endl;
    // pcout << "current max Cr number: " << max_cfl_number * time_step << std::endl;
    double time_output = 0.1;
    while (time < time_end + 1e-12)
    {
        ++timestep_number;
        pcout << "time step " << timestep_number << std::endl;
        run_once();
        time += time_step;
        if (time >= time_output)
        {
            output_image(outputDir, timestep_number, time);
            time_output += 0.1;
        }
        // if (timestep_number % 20 == 0)
        //     output_image("Sphere_Re3200_t0001_conform_refine3/", timestep_number);
        print_error();

        // current_cfl_number = get_cfl_number();
        // max_cfl_number = std::max(max_cfl_number, current_cfl_number);
        // pcout << "current Cr number: " << current_cfl_number * time_step << std::endl;
        // pcout << "current max Cr number: " << max_cfl_number * time_step << std::endl;

        computing_timer.print_summary();
        computing_timer.reset();
    }
}

template <int dim>
void GePUP<dim>::run_adaptive()
{
    unsigned int min_grid_level = n_global_refine;
    unsigned int max_grid_level = n_global_refine + 3;
    unsigned int n_initial_adaptive_refine = (max_grid_level - n_global_refine) * 2;
    double courant_number = 0.3;
    double max_timestep = 0.01;

    std::string output_image_name = "lid-driven-cavity/"; // 图片文件名称

    std::ofstream dof_output("lid-driven-cavity"); // 信息文件名称

    make_grid();
    setup_dofs();
    initialize_u();
    time_step = std::min(courant_number / get_cfl_number(), max_timestep);
    w_matrix.evaluate_coefficient();
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        w_mg_matrices[level].evaluate_coefficient();
    pcout << "      time step 0" << std::endl;
    pcout << "initial time_step: " << time_step << std::endl;
    output_image(output_image_name, 0, 0);

    // if (this_mpi_process == 0)
    //     dof_output << "time step " << timestep_number << "\tcurrent time " << time << "\tnumber of dofs " << dof_handler.n_dofs() << std::endl;

    print_error();
    ++timestep_number;
    pcout << "      time step " << timestep_number << std::endl;
    for (unsigned int pre_refine_step = 0; pre_refine_step < n_initial_adaptive_refine; ++pre_refine_step)
    {
        pcout << "      pre refine step " << pre_refine_step + 1 << std::endl;
        time = 0;
        initialize_u();
        run_once();
        time += time_step;
        // output_image(output_image_name, pre_refine_step + 1);
        pcout << "current time: " << time << std::endl;
        pcout << "time_step: " << time_step << std::endl;
        // print_error();

        auto time_map = computing_timer.get_summary_data(TimerOutput::OutputData::total_wall_time);
        double wall_time = time_map["setup dofs"] + time_map["compute old_w_integral"] + time_map["compute q and get minus_u_dot_grad_u_integral"] + time_map["compute ex_integral and laplacian_w_integral"] + time_map["compute w"] + time_map["compute phi"] + time_map["compute u"] + time_map["compute w_star"];

        // if (this_mpi_process == 0)
        //     dof_output << "prerefine step " << pre_refine_step + 1 << "\tcurrent time " << time << "\tnumber of dofs " << dof_handler.n_dofs() << "\tcomputing time " << wall_time << std::endl;

        computing_timer.print_summary();
        computing_timer.reset();

        refine_mesh(min_grid_level, max_grid_level);
        time_step = std::min(courant_number / get_cfl_number(), max_timestep);
        w_matrix.evaluate_coefficient();
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            w_mg_matrices[level].evaluate_coefficient();
    }
    double time_output = 0.1;
    while (time < time_end + 1e-12)
    {
        ++timestep_number;
        pcout << "      time step " << timestep_number << std::endl;
        run_once();
        time += time_step;
        pcout << "current time: " << time << std::endl;
        pcout << "time_step: " << time_step << std::endl;

        // output_image(output_image_name, timestep_number);
        if (time >= time_output)
        {
            output_image(output_image_name, timestep_number, time);
            time_output += 0.1;
        }
        print_error();

        auto time_map = computing_timer.get_summary_data(TimerOutput::OutputData::total_wall_time);
        double wall_time = time_map["setup dofs"] + time_map["compute old_w_integral"] + time_map["compute q and get minus_u_dot_grad_u_integral"] + time_map["compute ex_integral and laplacian_w_integral"] + time_map["compute w"] + time_map["compute phi"] + time_map["compute u"] + time_map["compute w_star"];

        // if (this_mpi_process == 0)
        //     dof_output << "time step " << timestep_number << "\t" << "current time " << time << "\t" << "number of dofs " << dof_handler.n_dofs() << "\tcomputing time " << wall_time << std::endl;

        computing_timer.print_summary();
        computing_timer.reset();

        // if (timestep_number % 5 == 0)//每五步调整网格
        refine_mesh(min_grid_level, max_grid_level);
        time_step = std::min(courant_number / get_cfl_number(), max_timestep);
        w_matrix.evaluate_coefficient();
        for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
            w_mg_matrices[level].evaluate_coefficient();
    }
}

int main(int argc, char **argv)
{
    unsigned int n_global_refine = 2;
    if (argc > 1)
        n_global_refine = atoi(argv[1]);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    GePUP<dimension> gepup(n_global_refine);
    gepup.run();
    // gepup.run_adaptive();
    return 0;
}