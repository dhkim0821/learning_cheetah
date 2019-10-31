#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>

int add(int i, int j) {
    return i + j;
}

Eigen::MatrixXd inv(const Eigen::MatrixXd &xs)
{
  return xs.inverse();
}

namespace py = pybind11;

class PointMass {
public:
    PointMass(const Eigen::Vector2d& goal) {
        this->reset();
        this->goal = goal;
    }

    void reset() {
        this->t = 0.0;
        this->dt = 0.05;
        this->m = 1.0;
        this->pos = Eigen::Vector2d::Zero();
        this->vel = Eigen::Vector2d::Zero();
    }

    void apply_force(const Eigen::Vector2d& f) {
        Eigen::Vector2d acc = f / this->m;
        const int num_internal_steps = 10;
        const float h = this->dt / num_internal_steps;

        for (int i = 0; i < num_internal_steps; ++i) {
            t += h;
            vel += h * acc;
            pos += h * vel;
        }
    }

    double m;
    double t;
    double dt;
    Eigen::Vector2d pos;
    Eigen::Vector2d vel;
    Eigen::Vector2d goal;
};

PYBIND11_MODULE(cenv, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cenv

        .. autosummary::
           :toctree: _generate

           add
           sub
           inv
           PointMass
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("sub", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("inv", &inv);

    py::class_<PointMass>(m, "PointMass")
        .def(py::init<const Eigen::Vector2d&>())
        .def("reset", &PointMass::reset)
        .def("apply_force", &PointMass::apply_force)
        .def_readwrite("m", &PointMass::m)
        .def_readwrite("t", &PointMass::t)
        .def_readwrite("dt", &PointMass::dt)
        .def_readwrite("pos", &PointMass::pos)
        .def_readwrite("vel", &PointMass::vel)
        .def_readwrite("goal", &PointMass::goal);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
