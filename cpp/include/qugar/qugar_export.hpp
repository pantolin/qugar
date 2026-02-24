
#ifndef QUGAR_EXPORT_H
#define QUGAR_EXPORT_H

#ifdef QUGAR_STATIC_DEFINE
#  define QUGAR_EXPORT
#  define QUGAR_NO_EXPORT
#else
#  ifndef QUGAR_EXPORT
#    ifdef qugar_EXPORTS
        /* We are building this library */
#      define QUGAR_EXPORT 
#    else
        /* We are using this library */
#      define QUGAR_EXPORT 
#    endif
#  endif

#  ifndef QUGAR_NO_EXPORT
#    define QUGAR_NO_EXPORT 
#  endif
#endif

#ifndef QUGAR_DEPRECATED
#  define QUGAR_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef QUGAR_DEPRECATED_EXPORT
#  define QUGAR_DEPRECATED_EXPORT QUGAR_EXPORT QUGAR_DEPRECATED
#endif

#ifndef QUGAR_DEPRECATED_NO_EXPORT
#  define QUGAR_DEPRECATED_NO_EXPORT QUGAR_NO_EXPORT QUGAR_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef QUGAR_NO_DEPRECATED
#    define QUGAR_NO_DEPRECATED
#  endif
#endif

#endif /* QUGAR_EXPORT_H */
