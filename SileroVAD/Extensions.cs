using System;
using System.Collections.Generic;

namespace SileroVAD
{
    internal static class Extensions
    {
        //Taken from https://stackoverflow.com/a/3210961/3733905
        internal static IEnumerable<T[]> Chunkify<T>(
            this IEnumerable<T> source, int size)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (size < 1) throw new ArgumentOutOfRangeException(nameof(size));
            using (var iter = source.GetEnumerator())
            {
                while (iter.MoveNext())
                {
                    var chunk = new T[size];
                    chunk[0] = iter.Current;
                    for (var i = 1; i < size && iter.MoveNext(); i++) chunk[i] = iter.Current;
                    yield return chunk;
                }
            }
        }
    }
}