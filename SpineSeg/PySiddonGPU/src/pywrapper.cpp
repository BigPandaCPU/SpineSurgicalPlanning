#include "siddon_class.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PySiddonGpu
{
	private:
		SiddonGpu *m_siddonGpu;
		int *m_numThreadsPerBlock;
		float *m_movImgArray;
		int *m_movSize;
		float *m_movSpacing;
		float m_X0, m_Y0, m_Z0;
		int *m_drrSize;
		
		float *m_source;
		float *m_destArray;
		float *m_drrArray;
		
	public:
		PySiddonGpu(py::array_t<int> numThreadsPerBlock, py::array_t<float> movImgArray, py::array_t<int> movSize, py::array_t<float> movSpacing, float X0, float Y0, float Z0, py::array_t<int> drrSize)
		{
			m_numThreadsPerBlock = static_cast<int*>(numThreadsPerBlock.request().ptr);
			m_movImgArray = static_cast<float*>(movImgArray.request().ptr);
			m_movSize = static_cast<int*>(movSize.request().ptr);
			m_movSpacing = static_cast<float*>(movSpacing.request().ptr);
			m_X0 = X0;
			m_Y0 = Y0;
			m_Z0 = Z0;
			m_drrSize = static_cast<int*>(drrSize.request().ptr);
			m_siddonGpu = new SiddonGpu(m_numThreadsPerBlock, m_movImgArray, m_movSize, m_movSpacing, m_X0, m_Y0, m_Z0, m_drrSize);
		}
		void releaseMem()
		{
			delete m_siddonGpu;
		}
		
		void generateDRR(py::array_t<float> source, py::array_t<float> destArray, py::array_t<float> drrArray)
		{
			m_source = static_cast<float*>(source.request().ptr);
			m_destArray = static_cast<float*>(destArray.request().ptr);
			m_drrArray = static_cast<float*>(drrArray.request().ptr);
			m_siddonGpu->generateDRR(m_source, m_destArray, m_drrArray);
		}
};

PYBIND11_MODULE(libPySiddonGpu, m) {
    py::class_<PySiddonGpu>(m, "PySiddonGpu")
      .def(py::init< py::array_t<int>, py::array_t<float>, py::array_t<int>, py::array_t<float>, float, float, float, py::array_t<int> >())
      .def("releaseMem",&PySiddonGpu::releaseMem)
      .def("generateDRR", [](PySiddonGpu& m_pySiddonGpu, py::array_t<float> source, py::array_t<float> destArray, py::array_t<float> drrArray){
          m_pySiddonGpu.generateDRR(source, destArray, drrArray);
      });
}