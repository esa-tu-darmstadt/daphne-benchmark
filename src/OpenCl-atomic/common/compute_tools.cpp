/**
 * Author:  Thilo Gabel, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2019
 * License: Apache 2.0 (see attachached File)
 */
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

#include "compute_tools.h"

ComputeEnv ComputeTools::find_compute_platform(
	std::string platformHint, std::string deviceHint, std::string deviceType,
	std::vector<std::vector<std::string>> extensions) {

	ComputeEnv result;

	// query all platforms
	std::vector<cl::Platform> availablePlatforms;
	try {
		cl::Platform::get(&availablePlatforms);
	} catch (cl::Error& e) {
		throw std::logic_error("Platform query failed: " + std::string(e.what()));
	}
	if (availablePlatforms.size() == 0) {
		throw std::logic_error("No platforms found");
	}
	// select a platform
	std::vector<cl::Platform> selectedPlatforms;
	if (platformHint.length() > 0) {
		// select certain platforms
		int iPlatform;
		if (sscanf(platformHint.c_str(), "%d", &iPlatform) == 1) {
			// select platform by index
			if (iPlatform < availablePlatforms.size()) {
				selectedPlatforms.push_back(availablePlatforms[iPlatform]);
			} else {
				throw std::logic_error("Platform of index" + std::to_string(iPlatform) + " does not exist");
			}

		} else {
			// search for platforms that match a given name
			bool found = false;
			for (cl::Platform p : availablePlatforms) {
				std::string platformName = p.getInfo<CL_PLATFORM_NAME>();
				if (platformName.find(platformHint) != std::string::npos) {
					selectedPlatforms.push_back(p);
					found = true;
				}
			}
			if (!found) {
				throw std::logic_error("No platform that matches " + platformHint);
			}
		}
	} else {
		// consider all platforms
		for (cl::Platform p : availablePlatforms) {
			selectedPlatforms.push_back(p);
		}
	}
	// query devices
	// filter devices by type
	std::vector<cl::Device> filteredDevices;
	// detect the device type
	cl_device_type type = CL_DEVICE_TYPE_ALL;
	if (deviceType.find("CPU") != std::string::npos) {
		type = CL_DEVICE_TYPE_CPU;
	} else if (deviceType.find("GPU") != std::string::npos) {
		type = CL_DEVICE_TYPE_GPU;
	} else if (deviceType.find("ACC") != std::string::npos) {
		type = CL_DEVICE_TYPE_ACCELERATOR;
	} else if (deviceType.find("DEF") != std::string::npos) {
		type = CL_DEVICE_TYPE_DEFAULT;
	}
	std::ostringstream sQueryError;
	bool errorDetected = false;
	// filter devices
	for (cl::Platform p : selectedPlatforms) {
		std::vector<cl::Device> devices;
		try {
			p.getDevices(type, &devices);
		} catch (cl::Error& e) {
			sQueryError << e.what() << " (" << e.err() << ")" << std::endl;
			errorDetected = true;
		}
		for (cl::Device d : devices) {
			filteredDevices.push_back(d);
		}
	}
	if (filteredDevices.size() == 0) {
		std::ostringstream sError;
		sError << "No devices found.";
		if (errorDetected) {
			sError << " Failed queries:" << std::endl;
			sError << sQueryError.str();
		}
		throw std::logic_error(sError.str());
	}
	// select devices
	std::vector<cl::Device> selectedDevices;
	if (deviceHint.length() > 0) {
		// select specific devices
		int iDevice;
		if (sscanf(deviceHint.c_str(), "%d", &iDevice) == 1) {
			// select by index
			if (iDevice < filteredDevices.size()) {
				selectedDevices.push_back(filteredDevices[iDevice]);
			} else {
				throw std::logic_error("Device of index " + std::to_string(iDevice) + " does not exist");
			}
		} else {
			// select by name
			bool found = false;
			for (cl::Device d : filteredDevices) {
				std::string deviceName = d.getInfo<CL_DEVICE_NAME>();
				if (deviceName.find(deviceHint) != std::string::npos) {
					selectedDevices.push_back(d);
					found = true;
				}
			}
			if (!found) {
				throw std::logic_error("No device that matches " + deviceHint);
			}
		}
	} else {
		// select all devices
		for (cl::Device d : filteredDevices) {
			selectedDevices.push_back(d);
		}
	}
	// filter by extensions
	std::vector<cl::Device> supportedDevices;
	if (extensions.size() > 0) {
		// request at least one extension
		bool found = false;
		for (cl::Device d : selectedDevices) {
			std::string supportedExtensions = d.getInfo<CL_DEVICE_EXTENSIONS>();
			// for each extension set at least one extension must be supported
			bool deviceSupported = true;
			for (std::vector<std::string> extensionSet : extensions) {
				bool extFound = false;
				for (std::string ext : extensionSet) {
					if (supportedExtensions.find(ext) != std::string::npos) {
						extFound = true;
					}
				}
				if (!extFound) {
					deviceSupported = false;
				}
			}
			if (deviceSupported) {
				supportedDevices.push_back(d);
			}
		}
		if (supportedDevices.size() == 0) {
			std::ostringstream sError;
			sError << "No device found that supports the required extensions: " << std::endl;
			for (std::vector<std::string> extensionSet : extensions) {
				sError << "{ ";
				for (std::string ext : extensionSet) {
					sError << ext << " ";
				}
				sError << "} ";
			}
			sError << std::endl;
			throw std::logic_error(sError.str());
		}
	} else {
		// all devices pass
		for (cl::Device d : selectedDevices) {
			supportedDevices.push_back(d);
		}
	}
	// create context and queue
	// select the first supported device
	result.device = supportedDevices[0];
	try {
		result.context = cl::Context(supportedDevices[0]);
	} catch (cl::Error& e) {
		throw std::logic_error("Context creation failed: " + std::string(e.what()));
	}
	try {
		result.cmdqueue = cl::CommandQueue(result.context, supportedDevices[0]);
	} catch (cl::Error& e) {
		throw std::logic_error("Command queue creation failed: " + std::string(e.what()));
	}
	return result;
}

cl::Program ComputeTools::build_program(ComputeEnv& computeEnv, std::string& sources,
	std::string options, std::vector<std::string>& kernelNames, std::vector<cl::Kernel>& kernels) {
	cl::Program::Sources sourcesCL;
	sourcesCL.push_back(std::make_pair(sources.c_str(), sources.size()));
	cl::Program program(computeEnv.context, sources);
	try {
		program.build(options.c_str());
	} catch (cl::Error& e) {
		std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(computeEnv.device);
		std::ostringstream sError;
		sError << "Failed to build program with flags: ";
		sError << options;
		sError << "(" << e.what() << "):" << std::endl;
		sError << log << std::endl;
		throw std::logic_error(sError.str());
	}
	kernels.clear();

	for (std::string name : kernelNames) {
		try {
			cl::Kernel kernel(program, name.c_str());
			kernels.push_back(kernel);
		} catch (cl::Error& e) {
			kernels.clear();
			std::ostringstream sError;
			sError << "Kernel " << name << " not found in program (";
			sError << e.what() << ")" << std::endl;
			throw std::logic_error(sError.str());
		}
	}
	return program;
}