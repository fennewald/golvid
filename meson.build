project('trnfr', 'cpp', 'cuda',
  default_options: ['warning_level=2', 'c_std=c23', 'cpp_std=gnu++23'])

if meson.get_compiler('cpp').get_linker_id() == 'ld.mold'
  # Enable colorful output if we're using mold
  add_global_link_arguments('-Wl,--color-diagnostics=always', language: 'cpp')
endif

add_global_arguments(['-Wno-pedantic'], language: 'c')

fmt_dep = subproject('fmt').get_variable('fmt_dep')

gst_dep = dependency('gstreamer-1.0', version: '>=1.20')

cuda_dep = dependency('cuda', version: '>=12')

srcs = [
  'src/main.cc',
  'src/sink.cc',
  'src/log/time.cc',
  'src/sim.cu'
]

exe = executable(
  'do',
  srcs,
  dependencies: [cuda_dep, gst_dep, fmt_dep]
)
