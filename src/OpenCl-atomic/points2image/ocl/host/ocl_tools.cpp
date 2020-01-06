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

#include "ocl_header.h"

OCL_Struct OCL_Tools::find_compute_platform(
	std::string platformHint, std::string deviceHint, std::string deviceType,
	std::vector<std::vector<std::string>> extensions) {

	OCL_Struct result;
	cl_int errorCode = CL_SUCCESS;
	// query all platforms
	std::vector<cl_platform_id> availablePlatforms(16);
	cl_uint availablePlatformNo = 0;
	errorCode = clGetPlatformIDs(availablePlatforms.size(), availablePlatforms.data(), &availablePlatformNo);
	if (errorCode != CL_SUCCESS) {
		throw std::logic_error("Platform query failed: " + errorCode);
	}
	availablePlatforms.resize(availablePlatformNo);
	if (availablePlatforms.size() == 0) {
		throw std::logic_error("No platforms found");
	}
	// select a platform
	std::vector<cl_platform_id> selectedPlatforms;
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
			for (cl_platform_id p : availablePlatforms) {
				std::vector<char> nameBuffer(256);
				size_t nameLength = 0;
				errorCode = clGetPlatformInfo(p, CL_PLATFORM_NAME, nameBuffer.size(), nameBuffer.data(), &nameLength);
				if (errorCode != CL_SUCCESS) {
					// ignore for now
				} else {
					std::string platformName = std::string(nameBuffer.data(), nameLength);
					if (platformName.find(platformHint) != std::string::npos) {
						selectedPlatforms.push_back(p);
						found = true;
					}
				}
			}
			if (!found) {
				throw std::logic_error("No platform that matches " + platformHint);
				// TODO: display any query errors
			}
		}
	} else {
		// consider all platforms
		for (cl_platform_id p : availablePlatforms) {
			selectedPlatforms.push_back(p);
		}
	}
	// query devices
	// filter devices by type
	std::vector<cl_device_id> filteredDevices;
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
	for (cl_platform_id p : selectedPlatforms) {
	std::vector<cl_device_id> devices(16);
	cl_uint deviceNo = 0;
	errorCode = clGetDeviceIDs(p, type, devices.size(), devices.data(), &deviceNo);
	if (errorCode != CL_SUCCESS) {
		sQueryError << "Failed to get devices (" << errorCode << ")" << std::endl;
		errorDetected = true;
	}
	devices.resize(deviceNo);
		for (cl_device_id d : devices) {
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
	std::vector<cl_device_id> selectedDevices;
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
		bool queryError = false;
		std::ostringstream sQueryError;
		for (cl_device_id d : filteredDevices) {
			std::vector<char> nameBuffer(256);
			size_t nameLength = 0;
			errorCode = clGetDeviceInfo(d, CL_DEVICE_NAME, nameBuffer.size(), nameBuffer.data(), &nameLength);
			if (errorCode != CL_SUCCESS) {
				sQueryError << "Failed to get device name (" << errorCode << ")" << std::endl;
				queryError = true;
			} else {
				std::string deviceName(nameBuffer.data(), nameLength);
				//std::string deviceName = d.getInfo<CL_DEVICE_NAME>();
				if (deviceName.find(deviceHint) != std::string::npos) {
					selectedDevices.push_back(d);
					found = true;
				}
			}
		}
		if (!found) {
			std::ostringstream sError;
			sError << "No device that matches " << deviceHint << std::endl;
			if (queryError) {
				sError << "Failed device queries:" << std::endl;
				sError << sQueryError.str();
			}
			throw std::logic_error(sError.str());
		}
	}
	} else {
		// select all devices
		for (cl_device_id d : filteredDevices) {
			selectedDevices.push_back(d);
		}
	}
	// filter by extensions
	std::vector<cl_device_id> supportedDevices;
	if (extensions.size() > 0) {
		// request at least one extension
		bool found = false;
		bool queryError = false;
		std::ostringstream sQueryError;
		for (cl_device_id d : selectedDevices) {
			std::vector<char> extensionBuffer(4096);
			size_t extensionLength = 0;
			errorCode = clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, extensionBuffer.size(), extensionBuffer.data(), &extensionLength);
			if (errorCode != CL_SUCCESS) {
				sQueryError << "Failed to get device extensions (" << errorCode << ")" << std::endl;
				queryError = true;
			} else {
				std::string supportedExtensions(extensionBuffer.data(), extensionLength);
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
			if (queryError) {
				sError << sQueryError.str();
			}
			throw std::logic_error(sError.str());
		}
	} else {
		// all devices pass
		for (cl_device_id d : selectedDevices) {
			supportedDevices.push_back(d);
		}
	}
	// create context and queue
	// select the first supported device
	result.device = supportedDevices[0];
	result.context = clCreateContext(nullptr, 1, &supportedDevices[0], nullptr, nullptr, &errorCode);
	if (errorCode != CL_SUCCESS) {
		throw std::logic_error("Context creation failed: " + std::to_string(errorCode));
	}
	cl_command_queue_properties queueProperties = 0;
	result.cmdqueue = clCreateCommandQueue(result.context, supportedDevices[0], queueProperties, &errorCode);
	if (errorCode != CL_SUCCESS) {
		throw std::logic_error("Command queue creation failed: " + std::to_string(errorCode));
	}
	return result;
}
cl_program OCL_Tools::build_program(OCL_Struct& ocl_objs, std::string& sources,
	std::string options, std::vector<std::string>& kernelNames, std::vector<cl_kernel>& kernels) {

	cl_int errorCode = CL_SUCCESS;
	const char* cSource = sources.c_str();
	cl_program program = clCreateProgramWithSource(ocl_objs.context, 1, &cSource, nullptr, &errorCode);
	if (errorCode != CL_SUCCESS) {
		throw std::logic_error("Failed to create program from source");
	}
	const char* cOptions = options.c_str();
	errorCode = clBuildProgram(program, 1, &ocl_objs.device, cOptions, NULL, NULL);
	if (errorCode != CL_SUCCESS) {
		std::vector<char> logBuffer(8192);
		size_t logLength = 0;
		errorCode = clGetProgramBuildInfo(program, ocl_objs.device,
			CL_PROGRAM_BUILD_LOG, logBuffer.size(), logBuffer.data(), &logLength);
		std::string log(logBuffer.data(), logLength);
		throw std::logic_error("Build failed:\n" + log);
	}
	kernels.clear();
	for (std::string name : kernelNames) {
		const char* cKernel = name.c_str();
		cl_kernel kernel = clCreateKernel(program, cKernel, &errorCode);
		if (errorCode == CL_SUCCESS) {
			kernels.push_back(kernel);
		} else {
			throw std::logic_error("Kernel " + name + " not found in program");
		}
	}
	return program;
}