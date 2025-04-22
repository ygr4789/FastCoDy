#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>

#include <linear_tetmesh_dphi_dX.h>
#include <linear_tetmesh_arap_dq.h>
#include <linear_tetmesh_arap_dq2.h>
#include <linear_tetmesh_arap_q.h>
#include <linear_tetmesh_neohookean_dq.h>
#include <linear_tetmesh_neohookean_dq2.h>
#include <linear_tetmesh_neohookean_q.h>
#include <linear_tetmesh_stvk_dq.h>
#include <linear_tetmesh_stvk_dq2.h>
#include <linear_tetmesh_stvk_q.h>
#include <linear_tetmesh_corotational_dq.h>
#include <linear_tetmesh_corotational_dq2.h>
#include <linear_tetmesh_corotational_q.h>
#include <simple_psd_fix.h>
#include <Eigen/Sparse>
#include <Eigen/Core>


namespace py = pybind11;
using SpMat = Eigen::SparseMatrix<double>;

/// Convert scipy.sparse.csr_matrix to Eigen::SparseMatrix
SpMat csr_to_eigen(const py::object& csr) {
    py::array data = csr.attr("data").cast<py::array>();
    py::array indices = csr.attr("indices").cast<py::array>();
    py::array indptr = csr.attr("indptr").cast<py::array>();
    std::pair<ssize_t, ssize_t> shape = csr.attr("shape").cast<std::pair<ssize_t, ssize_t>>();

    ssize_t rows = shape.first;
    ssize_t cols = shape.second;

    SpMat mat(rows, cols);
    std::vector<Eigen::Triplet<double>> triplets;

    auto p_data = data.unchecked<double, 1>();
    auto p_indices = indices.unchecked<int, 1>();
    auto p_indptr = indptr.unchecked<int, 1>();

    for (ssize_t row = 0; row < rows; ++row) {
        for (ssize_t idx = p_indptr(row); idx < p_indptr(row + 1); ++idx) {
            triplets.emplace_back(row, p_indices(idx), p_data(idx));
        }
    }

    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

/// Convert Eigen::SparseMatrix to scipy.sparse.csr_matrix
py::object eigen_to_csr(SpMat& mat) {
    mat.makeCompressed();

    auto data = py::array_t<double>(mat.nonZeros(), mat.valuePtr());
    auto indices = py::array_t<int>(mat.nonZeros(), mat.innerIndexPtr());
    auto indptr = py::array_t<int>(mat.rows() + 1, mat.outerIndexPtr());

    py::object scipy_sparse = py::module_::import("scipy.sparse");
    return scipy_sparse.attr("csr_matrix")(
        py::make_tuple(data, indices, indptr),
        py::arg("shape") = py::make_tuple(mat.rows(), mat.cols())
    );
}

namespace py = pybind11;

double linear_tetmesh_arap_q_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> V,
    py::array_t<int, py::array::c_style | py::array::forcecast> E,
    py::array_t<double, py::array::c_style | py::array::forcecast> q,
    py::array_t<double, py::array::c_style | py::array::forcecast> dphidX,
    py::array_t<double, py::array::c_style | py::array::forcecast> volume,
    py::array_t<double, py::array::c_style | py::array::forcecast> params
) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> V_mat(V.data(), V.shape(0), V.shape(1));
    Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> E_mat(E.data(), E.shape(0), E.shape(1));
    Eigen::Map<const Eigen::VectorXd> q_vec(q.data(), q.size());
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dphidX_mat(dphidX.data(), dphidX.shape(0), dphidX.shape(1));
    Eigen::Map<const Eigen::VectorXd> volume_vec(volume.data(), volume.shape(0));
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> params_mat(params.data(), params.shape(0), params.shape(1));
    
    // Call the function
    return sim::linear_tetmesh_arap_q(V_mat, E_mat, q_vec, dphidX_mat, volume_vec, params_mat);
}

py::array_t<double> linear_tetmesh_arap_dq_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> V,
    py::array_t<int, py::array::c_style | py::array::forcecast> E,
    py::array_t<double, py::array::c_style | py::array::forcecast> q,
    py::array_t<double, py::array::c_style | py::array::forcecast> dphidX,
    py::array_t<double, py::array::c_style | py::array::forcecast> volume,
    py::array_t<double, py::array::c_style | py::array::forcecast> params
) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> V_mat(V.data(), V.shape(0), V.shape(1));
    Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>  E_mat(E.data(), E.shape(0), E.shape(1));
    Eigen::Map<const Eigen::VectorXd> q_vec(q.data(), q.size());
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dphidX_mat(dphidX.data(), dphidX.shape(0), dphidX.shape(1));
    Eigen::Map<const Eigen::VectorXd> volume_vec(volume.data(), volume.shape(0));
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> params_mat(params.data(), params.shape(0), params.shape(1));
    Eigen::VectorXd result;
    
    // Call the function
    sim::linear_tetmesh_arap_dq(result, V_mat, E_mat, q_vec, dphidX_mat, volume_vec, params_mat);

    // Return result as numpy array
    return py::array_t<double>(
        {result.size()},  // shape
        {sizeof(double)}, // stride
        result.data()     // data pointer
    );
}

py::object linear_tetmesh_arap_dq2_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> V,
    py::array_t<int, py::array::c_style | py::array::forcecast> E,
    py::array_t<double, py::array::c_style | py::array::forcecast> q,
    py::array_t<double, py::array::c_style | py::array::forcecast> dphidX,
    py::array_t<double, py::array::c_style | py::array::forcecast> volume,
    py::array_t<double, py::array::c_style | py::array::forcecast> params
) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> V_mat(V.data(), V.shape(0), V.shape(1));
    Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> E_mat(E.data(), E.shape(0), E.shape(1));
    Eigen::Map<const Eigen::VectorXd> q_vec(q.data(), q.size());
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dphidX_mat(dphidX.data(), dphidX.shape(0), dphidX.shape(1));
    Eigen::Map<const Eigen::VectorXd> volume_vec(volume.data(), volume.shape(0));
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> params_mat(params.data(), params.shape(0), params.shape(1));
    SpMat result;
    
    // Call the function
    sim::linear_tetmesh_arap_dq2(result, V_mat, E_mat, q_vec, dphidX_mat, volume_vec, params_mat, [](auto &a) {sim::simple_psd_fix(a, 1e-3);});
    return eigen_to_csr(result);
}

py::array_t<double> linear_tetmesh_dphi_dX_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> V,
    py::array_t<int, py::array::c_style | py::array::forcecast> E
) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> V_mat(V.data(), V.shape(0), V.shape(1));
    Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> E_mat(E.data(), E.shape(0), E.shape(1));
    Eigen::MatrixXd result;

    // Call the function
    sim::linear_tetmesh_dphi_dX(result, V_mat, E_mat);

    // Return result as numpy array
    return py::array_t<double>(
        {result.rows(), result.cols()},                   // shape
        {sizeof(double), sizeof(double) * result.rows()}, // strides
        result.data()                                     // data pointer
    );
}

PYBIND11_MODULE(pyBartels, m) {
    m.def("linear_tetmesh_dphi_dX", &linear_tetmesh_dphi_dX_wrapper,
          "Compute dphi/dX for tetrahedral mesh");

    m.def("linear_tetmesh_arap_q", &linear_tetmesh_arap_q_wrapper,
          "Compute ARAP energy for tetrahedral mesh");
          
    m.def("linear_tetmesh_arap_dq", &linear_tetmesh_arap_dq_wrapper,
          "Compute ARAP energy gradient for tetrahedral mesh");
          
    m.def("linear_tetmesh_arap_dq2", &linear_tetmesh_arap_dq2_wrapper,
          "Compute ARAP energy gradient for tetrahedral mesh");
}