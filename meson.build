project('trnfr', 'cpp', 'cuda',
  default_options: ['warning_level=3', 'c_std=c23', 'cpp_std=c++23'])

if meson.get_compiler('cpp').get_linker_id() == 'ld.mold'
  # Enable colorful output if we're using mold
  add_global_link_arguments('-Wl,--color-diagnostics=always', language: 'cpp')
endif

fmt_dep = subproject('fmt').get_variable('fmt_dep')

gst_dep = dependency('gstreamer-1.0', version: '>=1.20')

srcs = [
  'src/main.cc',
  # Gstreamer
  'src/gst/init.cc',
  'src/gst/element.cc',
  'src/gst/pipeline.cc',
  'src/gst/bus.cc',
  'src/gst/bin.cc',
]

exe = executable(
  'do',
  srcs,
  dependencies: [gst_dep, fmt_dep]
)
