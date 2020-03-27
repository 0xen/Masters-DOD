#include "rendering.hpp"

#include <stdexcept>
#include <assert.h>


std::unique_ptr<Renderer> Renderer::sm_singlton = nullptr;
