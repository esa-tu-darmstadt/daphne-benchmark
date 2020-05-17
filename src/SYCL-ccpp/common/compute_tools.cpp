/**
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#include <stdexcept>

#include <SYCL/sycl.hpp>

#include "common/compute_tools.h"



// cl::sycl::device SyclTools::findComputeDevice(const std::string& deviceType) {
// 	cl::sycl::device result;
// 	if (deviceType.compare("CPU") == 0) {
// 		result = cl::sycl::cpu_selector().select_device();
// 	} else if (deviceType.compare("GPU") == 0) {
// 		result = cl::sycl::gpu_selector().select_device();
// 	} else if (deviceType.compare("HOST") == 0) {
// 		result = cl::sycl::host_selector().select_device();
// 	} else {
// 		result = cl::sycl::host_selector().select_device();
// 	}
// 	return result;
// }
ComputeEnv ComputeTools::find_compute_platform(
		const std::string& platformHint, const std::string& deviceHint, const std::string& deviceTypeHint) {
	// find platforms
	std::vector<cl::sycl::platform> availablePlatforms = cl::sycl::platform::get_platforms();
	if (availablePlatforms.size() == 0) {
		throw std::logic_error("No platforms found");
	}
	// filter by platform name
	std::vector<cl::sycl::platform> selectedPlatforms;
	if (platformHint.size() == 0) {
		for (cl::sycl::platform p : availablePlatforms) {
			selectedPlatforms.push_back(p);
		}
	} else {
		for (cl::sycl::platform p : availablePlatforms) {
			std::string platformName = p.get_info<cl::sycl::info::platform::name>();
			if (platformName.find(platformHint) != std::string::npos) {
				selectedPlatforms.push_back(p);
			}
		}
	}
	if (selectedPlatforms.size() == 0) {
		throw std::logic_error("No platforms named " + platformHint);
	}
	cl::sycl::info::device_type deviceType = cl::sycl::info::device_type::all;
	// find devices
	if (deviceTypeHint.compare("CPU") == 0) {
		deviceType = cl::sycl::info::device_type::cpu;
	} else if (deviceTypeHint.compare("GPU") == 0) {
		deviceType = cl::sycl::info::device_type::gpu;
	} else if (deviceTypeHint.compare("HOST") == 0) {
		deviceType = cl::sycl::info::device_type::host;
	}
	std::vector<cl::sycl::device> availableDevices;
	for (cl::sycl::platform p : selectedPlatforms) {
		std::vector<cl::sycl::device> platformDevices = p.get_devices(deviceType);
		for (cl::sycl::device d : platformDevices) {
			availableDevices.push_back(d);
		}
	}
	if (availableDevices.size() == 0) {
		throw std::logic_error("No " + deviceTypeHint + " devices available");
	}
	// filter by device name
	std::vector<cl::sycl::device> selectedDevices;
	if (deviceHint.size() == 0) {
		for (cl::sycl::device d : availableDevices) {
			selectedDevices.push_back(d);
		}
	} else {
		for (cl::sycl::device d : availableDevices) {
			std::string deviceName = d.get_info<cl::sycl::info::device::name>();
			if (deviceName.find(deviceHint) != std::string::npos) {
				selectedDevices.push_back(d);
			}
		}
	}
	// return the first device
	if (selectedDevices.size() == 0) {
		throw std::logic_error("No device named " + deviceHint);
	}
	cl::sycl::queue cmdqueue(selectedDevices[0]);
	ComputeEnv result = {
		selectedDevices[0],
		cmdqueue
	};
	return result;
}